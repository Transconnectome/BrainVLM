"""
UMBRELLA: Patch Embedding Module for 3D/4D Brain MRI

This module implements the PatchEmbed class for converting 3D T1-weighted MRI
and 4D rsfMRI volumes into patch embeddings for the LLaVA-based architecture.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for 3D T1 and 4D rsfMRI volumes.

    This module converts volumetric brain images into sequences of patch embeddings
    suitable for transformer-based processing. It supports both:
    - 3D T1-weighted structural MRI (shape: B x C x D x H x W)
    - 4D rsfMRI functional MRI (shape: B x C x D x H x W x T)

    Args:
        T1_size: Size of T1 images [D, H, W]
        T1_patch_size: Patch size for T1 images [pD, pH, pW]
        rsfMRI_size: Size of rsfMRI volumes [D, H, W, T]
        rsfMRI_patch_size: Patch size for rsfMRI [pD, pH, pW, pT]
        in_chans: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 768)
        dtype: Data type for parameters
        joint_optimization: If True, assert T1 and rsfMRI have same patch count
    """

    def __init__(self,
                 T1_size=[128, 128, 128],
                 T1_patch_size=[18, 18, 18],
                 rsfMRI_size=[96, 96, 96, 20],
                 rsfMRI_patch_size=[16, 16, 16, 5],
                 in_chans=1,
                 embed_dim=768,
                 dtype=torch.float32,
                 joint_optimization=False):
        super().__init__()
        self.embed_dim = embed_dim

        # Patchifying layer for T1 images
        T1_num_patches = (T1_size[0] // T1_patch_size[0]) * (T1_size[1] // T1_patch_size[1]) * (T1_size[2] // T1_patch_size[2])
        self.T1_grid_size = (T1_size[0] // T1_patch_size[0], T1_size[1] // T1_patch_size[1], T1_size[2] // T1_patch_size[2])
        self.T1_size = T1_size
        self.T1_patch_size = T1_patch_size
        self.T1_num_patches = T1_num_patches
        self.T1_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=T1_patch_size, stride=T1_patch_size, dtype=dtype)
        self.T1_positional_embeddings = nn.Parameter(torch.zeros(1, T1_num_patches, embed_dim))
        trunc_normal_(self.T1_positional_embeddings, std=.02)

        # Patchifying layer for rsfMRI
        rsfMRI_num_patches = (rsfMRI_size[0] // rsfMRI_patch_size[0]) * (rsfMRI_size[1] // rsfMRI_patch_size[1]) * (rsfMRI_size[2] // rsfMRI_patch_size[2]) * (rsfMRI_size[3] // rsfMRI_patch_size[3])
        self.rsfMRI_grid_size = (rsfMRI_size[0] // rsfMRI_patch_size[0], rsfMRI_size[1] // rsfMRI_patch_size[1], rsfMRI_size[2] // rsfMRI_patch_size[2], rsfMRI_size[3] // rsfMRI_patch_size[3])
        self.rsfMRI_size = rsfMRI_size
        self.rsfMRI_patch_size = rsfMRI_patch_size
        self.rsfMRI_num_patches = rsfMRI_num_patches
        self.rsfMRI_proj = nn.Linear(in_features=in_chans * rsfMRI_patch_size[0] * rsfMRI_patch_size[1] * rsfMRI_patch_size[2] * rsfMRI_patch_size[3], out_features=embed_dim)
        self.rsfMRI_positional_embeddings = nn.Parameter(torch.zeros(1, rsfMRI_num_patches, embed_dim))
        trunc_normal_(self.rsfMRI_positional_embeddings, std=.02)

        if joint_optimization:
            assert self.T1_num_patches == self.rsfMRI_num_patches, \
                f"For joint optimization, T1 patches ({self.T1_num_patches}) must equal rsfMRI patches ({self.rsfMRI_num_patches})"


    def forward_embeddings(self, x):
        """Process a single modality input into patch embeddings.

        Args:
            x: Input tensor, either 5D (T1) or 6D (rsfMRI)

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        if len(x.shape) == 5:
            # T1 MRI: (B, C, D, H, W)
            B, C, D, H, W = x.shape
            assert D == self.T1_size[0] and H == self.T1_size[1] and W == self.T1_size[2], \
                f"Input image size ({D}*{H}*{W}) doesn't match model ({self.T1_size[0]}*{self.T1_size[1]}*{self.T1_size[2]})."
            x = self.T1_proj(x).flatten(2).transpose(1, 2)  # B L C
            x = x + self.T1_positional_embeddings

        elif len(x.shape) == 6:
            # rsfMRI: (B, C, D, H, W, T)
            B, C, D, H, W, T = x.shape
            pD, pH, pW, pT = self.rsfMRI_grid_size
            sD, sH, sW, sT = self.rsfMRI_patch_size

            x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
            x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sD * sH * sW * sT * C)
            x = self.rsfMRI_proj(x)
            x = x.view(B, pD, pH, pW, -1, self.embed_dim).contiguous()
            x = x.permute(0, 5, 1, 2, 3, 4)
            x = x.flatten(2).transpose(1, 2)  # B L C
            x = x + self.rsfMRI_positional_embeddings

        return x

    def forward(self, x, interpolate_pos_encoding=False):
        """Forward pass for patch embedding.

        Args:
            x: Input tensor or dict of tensors keyed by modality
            interpolate_pos_encoding: Unused, for API compatibility

        Returns:
            Patch embeddings tensor
        """
        if isinstance(x, dict):
            modalities = list(x.keys())
            outputs = []
            for modality in modalities:
                embeddings = self.forward_embeddings(x[modality])
                outputs.append(embeddings)
            outputs = torch.cat(outputs, dim=0)
        else:
            outputs = self.forward_embeddings(x)

        return outputs
