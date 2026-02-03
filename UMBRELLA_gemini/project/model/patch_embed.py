"""
UMBRELLA: Patch Embedding Module for 3D/4D Brain MRI

This module implements the PatchEmbed class for converting 3D T1-weighted MRI
and 4D fMRI volumes into patch embeddings for the LLaVA-based architecture.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for 3D sMRI and 4D fMRI volumes.

    This module converts volumetric brain images into sequences of patch embeddings
    suitable for transformer-based processing. It supports both:
    - 3D T1-weighted structural MRI (shape: B x C x D x H x W)
    - 4D fMRI functional MRI (shape: B x C x D x H x W x T)

    Args:
        sMRI_size: Size of sMRI images [D, H, W]
        sMRI_patch_size: Patch size for sMRI images [pD, pH, pW]
        fMRI_size: Size of fMRI volumes [D, H, W, T]
        fMRI_patch_size: Patch size for fMRI [pD, pH, pW, pT]
        in_chans: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 768)
        dtype: Data type for parameters
        joint_optimization: If True, assert sMRI and fMRI have same patch count
    """

    def __init__(self,
                 sMRI_size=[128, 128, 128],
                 sMRI_patch_size=[18, 18, 18],
                 fMRI_size=[96, 96, 96, 20],
                 fMRI_patch_size=[16, 16, 16, 5],
                 in_chans=1,
                 embed_dim=768,
                 dtype=torch.float32,
                 joint_optimization=False):
        super().__init__()
        self.embed_dim = embed_dim

        # Patchifying layer for sMRI images
        sMRI_num_patches = (sMRI_size[0] // sMRI_patch_size[0]) * (sMRI_size[1] // sMRI_patch_size[1]) * (sMRI_size[2] // sMRI_patch_size[2])
        self.sMRI_grid_size = (sMRI_size[0] // sMRI_patch_size[0], sMRI_size[1] // sMRI_patch_size[1], sMRI_size[2] // sMRI_patch_size[2])
        self.sMRI_size = sMRI_size
        self.sMRI_patch_size = sMRI_patch_size
        self.sMRI_num_patches = sMRI_num_patches
        self.sMRI_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=sMRI_patch_size, stride=sMRI_patch_size, dtype=dtype)
        self.sMRI_positional_embeddings = nn.Parameter(torch.zeros(1, sMRI_num_patches, embed_dim))
        trunc_normal_(self.sMRI_positional_embeddings, std=.02)

        # Patchifying layer for fMRI
        fMRI_num_patches = (fMRI_size[0] // fMRI_patch_size[0]) * (fMRI_size[1] // fMRI_patch_size[1]) * (fMRI_size[2] // fMRI_patch_size[2]) * (fMRI_size[3] // fMRI_patch_size[3])
        self.fMRI_grid_size = (fMRI_size[0] // fMRI_patch_size[0], fMRI_size[1] // fMRI_patch_size[1], fMRI_size[2] // fMRI_patch_size[2], fMRI_size[3] // fMRI_patch_size[3])
        self.fMRI_size = fMRI_size
        self.fMRI_patch_size = fMRI_patch_size
        self.fMRI_num_patches = fMRI_num_patches
        self.fMRI_proj = nn.Linear(in_features=in_chans * fMRI_patch_size[0] * fMRI_patch_size[1] * fMRI_patch_size[2] * fMRI_patch_size[3], out_features=embed_dim)
        self.fMRI_positional_embeddings = nn.Parameter(torch.zeros(1, fMRI_num_patches, embed_dim))
        trunc_normal_(self.fMRI_positional_embeddings, std=.02)

        if joint_optimization:
            assert self.sMRI_num_patches == self.fMRI_num_patches, \
                f"For joint optimization, sMRI patches ({self.sMRI_num_patches}) must equal fMRI patches ({self.fMRI_num_patches})"


    def forward_embeddings(self, x):
        """Process a single modality input into patch embeddings.

        Args:
            x: Input tensor, either 5D (sMRI) or 6D (fMRI)

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        if len(x.shape) == 5:
            # sMRI MRI: (B, C, D, H, W)
            B, C, D, H, W = x.shape
            assert D == self.sMRI_size[0] and H == self.sMRI_size[1] and W == self.sMRI_size[2], \
                f"Input image size ({D}*{H}*{W}) doesn't match model ({self.sMRI_size[0]}*{self.sMRI_size[1]}*{self.sMRI_size[2]})."
            x = self.sMRI_proj(x).flatten(2).transpose(1, 2)  # B L C
            x = x + self.sMRI_positional_embeddings

        elif len(x.shape) == 6:
            # fMRI: (B, C, D, H, W, T)
            B, C, D, H, W, T = x.shape
            pD, pH, pW, pT = self.fMRI_grid_size
            sD, sH, sW, sT = self.fMRI_patch_size

            x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
            x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sD * sH * sW * sT * C)
            x = self.fMRI_proj(x)
            x = x.view(B, pD, pH, pW, -1, self.embed_dim).contiguous()
            x = x.permute(0, 5, 1, 2, 3, 4)
            x = x.flatten(2).transpose(1, 2)  # B L C
            x = x + self.fMRI_positional_embeddings

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
