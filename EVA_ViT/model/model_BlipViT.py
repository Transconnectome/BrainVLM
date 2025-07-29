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
        out_channels=2
    ):
        super().__init__()
        # setting model
        
        self.model = load_model(name=model_arch , model_type=model_type, is_eval=True, device='cpu')
        patch_embed_3d = PatchEmbed(
            img_size=img_size, 
            patch_size=self.model.visual_encoder.patch_embed.proj.kernel_size[0], 
            in_chans=1, 
            embed_dim=int(self.model.visual_encoder.patch_embed.proj.out_channels))
        num_patches = patch_embed_3d.num_patches
        pos_embed_3d = nn.Parameter(torch.zeros(1, num_patches + 1, int(self.model.visual_encoder.patch_embed.proj.out_channels)))
        trunc_normal_(pos_embed_3d, std=.02)

        # normalization and linear projection layer 
        self.norm = nn.LayerNorm(self.model.ln_vision.normalized_shape[0], eps=1e-6)
        self.proj = nn.Linear(self.model.ln_vision.normalized_shape[0], 2)
        trunc_normal_(self.proj.weight, std=.02)

        setattr(self.model.visual_encoder, "patch_embed", patch_embed_3d)
        setattr(self.model.visual_encoder,"pos_embed", pos_embed_3d)


        delattr(self.model, "t5_model")
        delattr(self.model, "Qformer")
        delattr(self.model, "ln_vision")
        
        # freeze every parameters except for patch embedding and positional embedding layer 
        for name, param in self.model.visual_encoder.named_parameters():
            if 'blocks' in name:
                param.requires_grad = False
            if 'cls_' in name: 
                param.requres_grad = False 
            if 'pos_embed' in name: 
                param.requires_grad = True 
            if 'patch_embed' in name: 
                param.requires_grad = True
    


    def forward(self, imgs, global_rank=None): 


        torch.cuda.empty_cache()
        image_emb = self.model.visual_encoder(imgs)
        image_emb = image_emb.mean(dim=1)
        image_emb = self.norm(image_emb)
        image_emb = self.proj(image_emb)

        return image_emb

