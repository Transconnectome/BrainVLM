"""
Multi-Image PatchEmbed for LLaVA-NeXT-Interleave style processing

This module supports processing multiple 3D brain MRI images independently,
similar to how LLaVA-NeXT-Interleave handles multiple 2D images.

Key differences from Bblip_t5.py:
- Supports batch dimension containing multiple images (e.g., reference + query)
- Each image is processed independently through the same patch embedding layer
- Returns concatenated features that can be interleaved in the language model
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class PatchEmbedInterleave(nn.Module):
    """
    Image to Patch Embedding with Multi-Image Support

    Supports processing multiple 3D brain MRI images where the batch dimension
    represents individual images that should be processed independently.

    Args:
        T1_size: Size of T1 image [D, H, W]
        T1_patch_size: Patch size for T1 [pD, pH, pW]
        in_chans: Number of input channels (default: 1)
        embed_dim: Embedding dimension
        dtype: Data type for parameters
    """

    def __init__(self,
                 T1_size=[120, 120, 120],
                 T1_patch_size=[10, 10, 10],
                 in_chans=1,
                 embed_dim=1152,  # SigLIP hidden_size for llava-interleave-qwen-0.5b-hf
                 dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim

        # Patchifying layer for T1 images
        T1_num_patches = (T1_size[0] // T1_patch_size[0]) * \
                        (T1_size[1] // T1_patch_size[1]) * \
                        (T1_size[2] // T1_patch_size[2])

        self.T1_grid_size = (
            T1_size[0] // T1_patch_size[0],
            T1_size[1] // T1_patch_size[1],
            T1_size[2] // T1_patch_size[2]
        )
        self.T1_size = T1_size
        self.T1_patch_size = T1_patch_size
        self.T1_num_patches = T1_num_patches

        # Convolutional projection layer
        self.T1_proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=T1_patch_size,
            stride=T1_patch_size,
            dtype=dtype
        )

        # Positional embeddings
        self.T1_positional_embeddings = nn.Parameter(
            torch.zeros(1, T1_num_patches, embed_dim)
        )
        trunc_normal_(self.T1_positional_embeddings, std=.02)


    def forward_embeddings(self, x):
        """
        Process 3D brain MRI through patch embedding

        Args:
            x: Input tensor of shape [B, C, D, H, W]
               B can be batch_size OR batch_size * num_images
               Each image along the B dimension is processed independently

        Returns:
            Patch embeddings of shape [B, num_patches, embed_dim]
        """
        if len(x.shape) == 5:
            B, C, D, H, W = x.shape

            # Validate input size
            assert D == self.T1_size[0] and H == self.T1_size[1] and W == self.T1_size[2], \
                f"Input image size ({D}*{H}*{W}) doesn't match model ({self.T1_size[0]}*{self.T1_size[1]}*{self.T1_size[2]})."

            # Apply convolutional projection
            # Input: [B, C, D, H, W]
            # Output: [B, embed_dim, grid_D, grid_H, grid_W]
            x = self.T1_proj(x)

            # Flatten spatial dimensions and transpose
            # [B, embed_dim, grid_D, grid_H, grid_W] -> [B, embed_dim, num_patches]
            x = x.flatten(2)

            # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
            x = x.transpose(1, 2)

            # Add positional embeddings
            # Positional embeddings are shared across all images in the batch
            x = x + self.T1_positional_embeddings

            return x  # [B, num_patches, embed_dim]
        else:
            raise ValueError(f"Expected 5D tensor [B, C, D, H, W], got shape {x.shape}")


    def forward(self, x, interpolate_pos_encoding=False):
        """
        Forward pass supporting both single and multi-image inputs

        Args:
            x: Input tensor, can be:
               - [B, C, D, H, W]: Standard batch of images
               - [1, num_images, C, D, H, W]: Batch with multiple images per sample
            interpolate_pos_encoding: Not used, for API compatibility

        Returns:
            Patch embeddings [B*num_images, num_patches, embed_dim]

        Note:
            Unlike the original PatchEmbed, this version does NOT concatenate
            embeddings from multiple images along the batch dimension.
            Instead, it keeps them separate so they can be interleaved properly
            with text tokens in the language model.
        """
        if isinstance(x, dict):
            # Handle dict input (multi-modality case)
            # For multi-turn comparison, we only use 'T1' modality
            raise NotImplementedError(
                "Multi-modality dict input not supported in PatchEmbedInterleave. "
                "Use separate forward passes for each modality."
            )
        else:
            # Check if input has extra batch dimension from data collator
            if len(x.shape) == 6:
                # Shape: [batch_size, num_images, C, D, H, W]
                # Reshape to: [batch_size * num_images, C, D, H, W]
                batch_size, num_images, C, D, H, W = x.shape
                x = x.reshape(batch_size * num_images, C, D, H, W)

            # Process all images in the batch
            outputs = self.forward_embeddings(x)
            return outputs


    def get_num_patches(self):
        """Return the number of patches per image"""
        return self.T1_num_patches
