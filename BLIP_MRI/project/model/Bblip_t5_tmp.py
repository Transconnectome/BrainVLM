# ref 1: https://github.com/Qybc/MedBLIP/blob/main/medblip/modeling_medblip_biomedlm.py
# ref 2: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
# ref 3: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py
# ref 4: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_t5_instruct.py
# ref 5: https://github.com/QwenLM/Qwen/blob/main/finetune.py#L172


import os 
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import torch.distributed as dist
#import lightning as L
import pytorch_lightning as pl

from lavis.models import load_model
from timm.models.layers import drop_path, to_3tuple, trunc_normal_
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
#from .eva_vit import create_eva_vit_g, PatchEmbed

from utils.utils import scaling_lr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

import loralib as lora
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from transformers import AutoTokenizer, Blip2ForConditionalGeneration
import types

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, 
                 T1_size=[128, 128, 128], 
                 T1_patch_size=[18, 18, 18], 
                 rsfmri_size=[96, 96, 96, 20],
                 rsfmri_patch_size=[16, 16, 16, 5],
                 in_chans=1, 
                 embed_dim=768, 
                 dtype=torch.float32):
        super().__init__()
        # patchifying layer for T1 images 
        T1_num_patches = (T1_size[0] // T1_patch_size[0]) * (T1_size[1] // T1_patch_size[1]) * (T1_size[2] // T1_patch_size[2])
        self.T1_grid_size = (T1_size[0] // T1_patch_size[0], T1_size[1] // T1_patch_size[1], T1_size[2] // T1_patch_size[2])
        self.T1_size = T1_size
        self.T1_patch_size = T1_patch_size
        self.T1_num_patches = T1_num_patches
        self.T1_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=T1_patch_size, stride=T1_patch_size, dtype=dtype)
        self.T1_positional_embeddings = nn.Parameter(torch.zeros(1, T1_num_patches, embed_dim))
        trunc_normal_(self.T1_positional_embeddings, std=.02)
        
        # patchifying layer for rsfmri
        rsfmri_num_patches = (rsfmri_size[0] // rsfmri_patch_size[0]) * (rsfmri_size[1] // rsfmri_patch_size[1]) * (rsfmri_size[2] // rsfmri_patch_size[2]) * (rsfmri_size[3] // rsfmri_patch_size[3]) 
        self.rsfmri_grid_size = (rsfmri_size[0] // rsfmri_patch_size[0], rsfmri_size[1] // rsfmri_patch_size[1], rsfmri_size[2] // rsfmri_patch_size[2], rsfmri_size[3] // rsfmri_patch_size[3])
        self.rsfmri_size = rsfmri_size
        self.rsfmri_patch_size = rsfmri_patch_size
        self.rsfmri_num_patches = rsfmri_num_patches
        self.rsfmri_proj = nn.Linear(in_features=in_chans * rsfmri_patch_size[0] * rsfmri_patch_size[1] * rsfmri_patch_size[2] * rsfmri_patch_size[3], out_features=embed_dim)
        self.rsfmri_positional_embeddings = nn.Parameter(torch.zeros(1, rsfmri_num_patches, embed_dim))
        trunc_normal_(self.rsfmri_positional_embeddings, std=.02)


    def forward(self, x, interpolate_pos_encoding=False):
        ##TODO 
        # get T1 and fMRI seperately and concat here at the end of this method

        if len(x.shape) == 5:
            B, C, D, H, W = x.shape
            # FIXME look at relaxing size constraints
            assert D == self.T1_size[0] and H == self.T1_size[1] and W == self.T1_size[2], \
                f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
            x = self.T1_proj(x).flatten(2).transpose(1, 2) # B L C 
            x = x + self.T1_positional_embeddings

        elif len(x.shape) == 6: 
            B, C, D, H, W, T = x.shape
            pD, pH, pW, pT = self.rsfmri_grid_size
            sD, sH, sW, sT = self.rsfmri_patch_size

            x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
            x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sD * sH * sW * sT * C)
            x = self.rsfmri_proj(x)
            x = x.view(B, pD, pH, pW, -1, self.embed_dim).contiguous()
            x = x.permute(0, 5, 1, 2, 3, 4)
            x = x.flatten(2).transpose(1, 2)    # B L C 
            x = x + self.rsfmri_positional_embeddings
        return x
    

