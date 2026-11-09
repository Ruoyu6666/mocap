import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap") 
from models.skeletonMAE.model.old.layers import MLP, DropPath
from models.skeletonMAE.model.layers import apply_rope


class Attention(nn.Module):
    """
    Attention + cross-attention: with `ctx`, queries come from x and keys/values from ctx,
    each rotated with its own positions (cos/sin for x, cos_c/sin_c for ctx).
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., rope_ratio=0.5, cross=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rot_dim = int(head_dim * rope_ratio) // 2 * 2  # keep even for chunking

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        if cross:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def rope(self, x, cos, sin):
        if cos is None or self.rot_dim == 0:
            return x
        x_rot, x_pass = x[..., :self.rot_dim], x[..., self.rot_dim:]
        return torch.cat([apply_rope(x_rot, cos, sin), x_pass], dim=-1)

    def forward(self, x, cos=None, sin=None, ctx=None, cos_c=None, sin_c=None):
        B, N, C = x.shape
        if ctx is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            cos_c, sin_c = cos, sin
        else:
            M = ctx.shape[1]
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            kv = self.kv(ctx).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        q = self.rope(q, cos, sin)
        k = self.rope(k, cos_c, sin_c)

        # same as softmax(q @ k^T * scale) -> attn_drop -> @ v, but with the fused kernel
        x = F.scaled_dot_product_attention(q, k, v, 
                    dropout_p=self.attn_drop.p if self.training else 0., scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 rope_ratio=0.5, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_ctx = norm_layer(dim) if cross else None
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, rope_ratio, cross)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, cos=None, sin=None, ctx=None, cos_c=None, sin_c=None):
        ctx = None if ctx is None else self.norm_ctx(ctx)
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin, ctx, cos_c, sin_c))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------------------------------------------------------- modules
class Tokenizer(nn.Module):
    """(B, T, J, 2) -> (B, T/patch_t, G, D); separate linear per joint group."""

    def __init__(self, n_joints, n_groups, patch_t, dim):
        super().__init__()
        self.pt, self.G = patch_t, n_groups
        self.jg = n_joints // n_groups
        c = self.pt * self.jg * 2
        self.w = nn.Parameter(torch.randn(self.G, c, dim) * c ** -0.5)
        self.b = nn.Parameter(torch.zeros(self.G, dim))

    def forward(self, x):
        B, T, _, _ = x.shape
        x = x.reshape(B, T // self.pt, self.pt, self.G, self.jg, 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, T // self.pt, self.G, -1)
        return torch.einsum("btgc,gcd->btgd", x, self.w) + self.b