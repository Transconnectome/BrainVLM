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

import loralib as lora




class Brain_BLIP(Blip2Base):
    def __init__(
        self,
        model_arch="blip2_opt",
        model_type="pretrain_opt2.7b",
        img_size=128,
        lora_vit=False, 
        lora_llm=False,
    ):
        super().__init__()
        # setting model
        
        self.model = load_model(name=model_arch , model_type=model_type, is_eval=True, device='cpu')
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
        """
        ## setting lm_head as lora layer
        lora_lm_head = lora.Linear(in_features=self.model.t5_model.lm_head.in_features, 
                                   out_features=self.model.t5_model.lm_head.out_features, 
                                   bias=self.model.t5_model.lm_head.bias,
                                   r=32)
        lora_lm_head.weight = self.model.t5_model.lm_head.weight
        lora_lm_head.bias = self.model.t5_model.lm_head.bias
        setattr(self.model.t5_model, "lm_head", lora_lm_head)
        lora.mark_only_lora_as_trainable(self.model)    # This sets requires_grad to False for all parameters without the string "lora_" in their names, thus it should be ran first for setting patching layer's requires_grad to True
        """

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
            if 'opt_proj' in name:
                param.requires_grad = False
        # freeze query token 
        for name, param in self.model.named_parameters():
            if 'query_tokens' in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'opt_model' in name:
                param.requires_grad = False
        for name, param in self.model.opt_model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = False
 
    def forward(self, batch, global_rank=None): 
        torch.cuda.empty_cache()
        #change the key name
        #batch['text_input'], batch['text_output'] = batch['inst'], batch['answer']
        #del batch['inst']
        #del batch['answer']
        loss_dict = self.model.forward(batch)
        pred = self.generate(batch)
        #pred = pred.detach().cpu().tolist()

        ### for sex classification
        #pred = [0 if sex == 'male' else 1 for sex in pred]
        ### for age classification
        try:
            pred = [float(value) for value in pred]
        except: 
            pass
    
        
        torch.cuda.empty_cache()
        return loss_dict['loss'], loss_dict, pred


    @torch.no_grad()
    def generate(
        self,
        batch,
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
        batch['prompt'] = batch['text_input'][0]
        #del batch['inst']
        output_text = self.model.generate(batch)
        #print(f"GT: {batch['answer']}\nPRED:{output_text}")
        return output_text


