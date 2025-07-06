""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .helpers import to_2tuple, to_3tuple
from .trace_utils import _assert


class PatchEmbed_2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed_3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        _assert(D == self.img_size[2], f"Input image depth ({D}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x
    


class PatchEmbed_EC_2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, in_chans=1, embed_dim=768, conv_dim=[64, 128, 256, 512], norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patch_size = to_2tuple(2 ** len(conv_dim))
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.ModuleList([])
        for i, _ in enumerate(conv_dim): 
            # the first convolution block
            if i == 0: 
                self.proj.append(nn.Conv2d(in_chans, conv_dim[i], kernel_size=3, stride=2, padding=1))
                self.proj.append(nn.BatchNorm2d(conv_dim[i]))
                self.proj.append(nn.ReLU(inplace=True))   
            # the intermediate convolution blocks
            else: 
                self.proj.append(nn.Conv2d(conv_dim[i-1], conv_dim[i], kernel_size=3, stride=2, padding=1))
                self.proj.append(nn.BatchNorm2d(conv_dim[i]))
                self.proj.append(nn.ReLU(inplace=True))
        # the last convolution block
        self.proj.append(nn.Conv2d(conv_dim[-1], embed_dim, kernel_size=1, stride=1, padding=0)) 

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        for layer in self.proj: 
            x = layer(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed_EC_3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, in_chans=1, embed_dim=768, conv_dim=[64, 128, 256, 512], norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_3tuple(img_size)
        self.img_size = img_size
        self.patch_size = to_3tuple(2 ** len(conv_dim))
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1], img_size[2] // self.patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.ModuleList([])
        for i, _ in enumerate(conv_dim): 
            # the first convolution block
            if i == 0: 
                self.proj.append(nn.Conv3d(in_chans, conv_dim[i], kernel_size=3, stride=2, padding=1))
                self.proj.append(nn.BatchNorm3d(conv_dim[i]))
                self.proj.append(nn.ReLU(inplace=True))   
            # the intermediate convolution blocks
            else: 
                self.proj.append(nn.Conv3d(conv_dim[i-1], conv_dim[i], kernel_size=3, stride=2, padding=1))
                self.proj.append(nn.BatchNorm3d(conv_dim[i]))
                self.proj.append(nn.ReLU(inplace=True))
        # the last convolution block
        self.proj.append(nn.Conv3d(conv_dim[-1], embed_dim, kernel_size=1, stride=1, padding=0))  
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        for layer in self.proj: 
            x = layer(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x
    

class PatchEmbed_CV_2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=[96, 96, 96], patch_size=16, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # BCHW -> norm(BHWC) -> BCHW
        x = self.act(x)
        return x


class PatchEmbed_CV_3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=[96, 96, 96], patch_size=16, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W, D = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        _assert(D == self.img_size[2], f"Input image depth ({D}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)   # BCHWD -> norm(BHWDC) -> BCHWD
        x = self.act(x)
        return x
    

