import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# RoPE / blocks
def rope_cos_sin(pos, head_dim, base):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=pos.device, dtype=torch.float32) / head_dim))
    ang = pos.float()[..., None] * inv  # (B, N, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, H, N, hd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[:, None], sin[:, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, rope_base=10000.0):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0
        self.heads, self.hd, self.base = heads, dim // heads, rope_base
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, pos):  # x (B, N, D), pos (B, N) in time-step units
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        cos, sin = rope_cos_sin(pos, self.hd, self.base)
        q = apply_rope(q, cos, sin).to(v.dtype)
        k = apply_rope(k, cos, sin).to(v.dtype)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))



class AttnPool(nn.Module):
    """One learned query attends over the tokens of a span (robust to different token counts)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):  # (N, K, D) -> (N, D)
        x = self.norm(x)
        return self.attn(self.q.expand(x.size(0), -1, -1), x, x, need_weights=False)[0][:, 0]