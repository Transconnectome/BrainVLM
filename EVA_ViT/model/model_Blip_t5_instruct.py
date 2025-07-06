import os 
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import torch.distributed as dist
import pytorch_lightning as pl

from lavis.models import load_model
from timm.models.layers import trunc_normal_
from lavis.models.blip2_models.blip2 import Blip2Base
from .model_EvaViT import PatchEmbed




class Brain_BLIP(Blip2Base):
    def __init__(
        self,
        model_arch="blip2_t5",
        model_type="pretrain_flant5xl",
        img_size=128,
        lora_vit=False, 
        lora_llm=False,
    ):
        super().__init__()
        # setting model
        
        self.model = load_model(name=model_arch , model_type=model_type, is_eval=True, device='cuda:0').to('cpu')
        self.model = self.model.half().float()
        patch_embed_3d = PatchEmbed(
            img_size=img_size, 
            #patch_size=self.model.visual_encoder.patch_embed.proj.kernel_size[0], 
            patch_size=18, #approximate of oringinal length of eva_clip g
            in_chans=1, 
            embed_dim=int(self.model.visual_encoder.patch_embed.proj.out_channels))
        num_patches = patch_embed_3d.num_patches
        pos_embed_3d = nn.Parameter(torch.zeros(1, num_patches + 1, int(self.model.visual_encoder.patch_embed.proj.out_channels)))
        trunc_normal_(pos_embed_3d, std=.02)

        # change patchify layer and positional embeddings
        setattr(self.model.visual_encoder, "patch_embed", patch_embed_3d)
        setattr(self.model.visual_encoder,"pos_embed", pos_embed_3d)

                
        for name, param in self.model.visual_encoder.named_parameters():
            if 'blocks' in name:
                param.requires_grad = False
            if 'cls_' in name: 
                param.requres_grad = False 
            if 'pos_embed' in name: 
                param.requires_grad = True 
            if 'patch_embed' in name: 
                param.requires_grad = True
        # freeze Qformer
        for name, param in self.model.named_parameters():
            if 'Qformer' in name:
                param.requires_grad = False
            if 't5_proj' in name:
                param.requires_grad = False
        # freeze query token 
        for name, param in self.model.named_parameters():
            if 'query_tokens' in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 't5_model' in name:
                param.requires_grad = False
        for name, param in self.model.t5_model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True


    def forward_revised(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        
        image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.model.qformer_text_input:
            text_Qformer = self.model.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.model.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.model.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        fs_embeds, fs_atts = None, None
        if self.model.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.model.prepare_few_shot_embeds(samples['few_shot_samples'])


        input_tokens = self.model.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
        output_tokens = self.model.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.model.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.model.t5_tokenizer.pad_token_id, -100
            )

        inputs_embeds = self.model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

        outputs = self.model.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
 
    def forward(self, batch, global_rank=None): 
        torch.cuda.empty_cache()
        #change the key name
        #batch['text_input'], batch['text_output'] = batch['inst'], batch['answer']
        #del batch['inst']
        #del batch['answer']
        #loss_dict = self.model.forward(batch)
        loss_dict = self.forward_revised(batch)
        pred = self.generate(batch)
        #pred = pred.detach().cpu().tolist()

        ### for sex classification
        #pred = [0 if sex == 'male' else 1 for sex in pred]
        ### for age classification
        try:
            pred = [int(age) for age in pred]
        except: 
            pass
    
        
        torch.cuda.empty_cache()
        return loss_dict['loss'], loss_dict, pred


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=5,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        device='cuda:0',
        ):
        samples['prompt'] = samples['text_input']
        #del batch['inst']
        
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.model.query_tokens.expand(bs, -1, -1)
        if self.model.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.model.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
        

        
        image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if self.model.qformer_text_input:
            query_output = self.model.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
            )
        else:
            query_output = self.model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
            )

        inputs_t5 = self.model.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.model.t5_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)


        inputs_embeds = self.model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.model.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
        )
        output_text = self.model.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
        )

        return output_text
        


