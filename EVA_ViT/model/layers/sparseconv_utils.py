import torch
import torch.nn as nn
from timm.models.layers import DropPath


_cur_active: torch.Tensor = None            # B1ff
# todo: try to use `gather` for speed?
def _get_active_ex_or_ii_2d(H, W, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def _get_active_ex_or_ii_3d(H, W, D, returning_active_ex=True):
    h_repeat, w_repeat, d_repeat = H // _cur_active.shape[-3], W // _cur_active.shape[-2], D // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3).repeat_interleave(d_repeat, dim=4)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi, di


def sp_conv_forward_2d(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii_2d(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_conv_forward_3d(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii_3d(H=x.shape[2], W=x.shape[3], D=x.shape[4],returning_active_ex=True)    # (BCHWD) *= (B1HWD), mask the output of conv
    return x


def sp_bn_forward_2d(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii_2d(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
    
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`
    
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


def sp_bn_forward_3d(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii_3d(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)

    bhwdc = x.permute(0, 2, 3, 4, 1)

    nc = bhwdc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`
    
    bchwd = torch.zeros_like(bhwdc)
    bchwd[ii] = nc
    bchwd = bchwd.permute(0, 4, 1, 2, 3)
    return bchwd



# 2D version
class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward_2d  # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseMaxPooling2d(nn.MaxPool2d):
    forward = sp_conv_forward_2d   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseAvgPooling2d(nn.AvgPool2d):
    forward = sp_conv_forward_2d   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward_2d     # hack: override the forward function; see `sp_bn_forward` above for more details

class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward_2d     # hack: override the forward function; see `sp_bn_forward` above for more details


# 3D version 
class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward_3d  # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseMaxPooling3d(nn.MaxPool3d):
    forward = sp_conv_forward_3d   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseAvgPooling3d(nn.AvgPool3d):
    forward = sp_conv_forward_3d   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseBatchNorm3d(nn.BatchNorm1d):
    forward = sp_bn_forward_3d     # hack: override the forward function; see `sp_bn_forward` above for more details

class SparseSyncBatchNorm3d(nn.SyncBatchNorm):
    forward = sp_bn_forward_3d     # hack: override the forward function; see `sp_bn_forward` above for more details