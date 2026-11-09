import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearStem(nn.Module):
    """
    (B, T, J, C) -> (B, T, D). Replace with any per-frame module of the same signature
    (e.g. a spatial-only joint encoder).
    """
    t_patch = 1
 
    def __init__(self, J, C, dim):
        super().__init__()
        self.proj = nn.Linear(J * C, dim)
        self.norm = nn.LayerNorm(dim)
 
    def forward(self, x, mask=None, mask_token=None):
        B, T, J, C = x.shape
        h = self.norm(self.proj(x.reshape(B, T, J * C)))
        if mask is not None:
            h = torch.where(mask[..., None], mask_token.to(h.dtype), h)
        return h
 
 
class WindowAttention(nn.Module):
    """Non-overlapping window self-attention along time. window<=0 -> one global window."""
    #  (B, T, D)  ->  (B, T, D)
    def __init__(self, dim, heads=4, window=16):
        super().__init__()
        assert dim % heads == 0
        self.h, self.window = heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
 
    def forward(self, x):                                   # (B,T,D)
        B, T, D = x.shape
        w = T if self.window <= 0 else min(self.window, T)
        pad = (-T) % w
        nW = (T + pad) // w
        xw = F.pad(x, (0, 0, 0, pad)).reshape(B * nW, w, D)
        qkv = self.qkv(xw).reshape(B * nW, w, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B*nW, H, w, Dh)
        mask = None
        
        if pad:                                             # hide padded keys in the last window
            valid = torch.arange(T + pad, device=x.device) < T
            mask = valid.reshape(1, nW, w).expand(B, -1, -1).reshape(B * nW, 1, 1, w)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B * nW, w, D)
        return self.out(o).reshape(B, nW * w, D)[:, :T]  # (B, T, D)
 


class Block(nn.Module):
    """conv pos-enc -> window attention -> dilated depthwise conv -> MLP (all residual)."""
 
    def __init__(self, dim, heads, window, dilation, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.cpe = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim, padding_mode="replicate")
        #  groups=2, equivalent to having two conv layers side by side, each seeing half the input channels 
        #            and producing half the output channels, and both subsequently concatenated
        #  groups = in_channgels,each input channel is convolved with its own set of filters 
        self.n1 = nn.LayerNorm(dim)
        
        self.attn = WindowAttention(dim, heads, window)
        self.n2 = nn.LayerNorm(dim)
        
        self.dw = nn.Conv1d(dim, dim, 3, dilation=dilation, padding=dilation, groups=dim, padding_mode="replicate")
        self.pw = nn.Linear(dim, dim)
        self.n3 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.drop = nn.Dropout(dropout)
 
    def forward(self, x):                                   # (B,T,D)
        x = x + self.cpe(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.attn(self.n1(x)))
        h = self.dw(self.n2(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.pw(F.gelu(h)))
        return x + self.drop(self.mlp(self.n3(x)))
 

 
class SpatialBlock(nn.Module):
    """Full attention across the joint-group tokens of ONE time step, then an MLP."""
 
    def __init__(self, dim, heads, mlp_ratio=2):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window=0)   # global over the G group tokens
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
 
    def forward(self, x):                                   # (N,G,D)
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))
 
 
class SkeleStem(nn.Module):
    """
    Group-aware tokenizer. (B,T,V,C) -> (B,T/t_patch,D).
      1. patchify: each (t_patch frames x p joints) patch -> one token per group, with a
         SEPARATE linear projection per joint group (no weight sharing across body parts)
      2. mask (time-step level, all groups) -> add learned group embedding
      3. n_blocks x [spatial attention across groups, temporal window attention per group]
      4. concatenate the G group tokens and project back to D (keeps part identity)
    groups: list of equal-size joint-index lists (anatomical groups); default = consecutive
            blocks of patch_size joints, so your keypoint order must keep related joints adjacent.
    """
 
    def __init__(self, dim, num_joints=12, dim_in=2, patch_size=3, t_patch=2, groups=None,
                 n_blocks=2, heads=4, window=16):
        super().__init__()
        
        if groups is None:
            assert num_joints % patch_size == 0, f"num_joints={num_joints} must be divisible by patch_size={patch_size}"
            groups = [list(range(g * patch_size, (g + 1) * patch_size)) for g in range(num_joints // patch_size)]
        assert len({len(g) for g in groups}) == 1, "all joint groups must have equal size"
        self.G, self.p = len(groups), len(groups[0])
        self.num_joints, self.dim_in, self.t_patch = num_joints, dim_in, t_patch
        self.register_buffer("idx", torch.tensor([j for g in groups for j in g]), persistent=False)
        
        fan_in = t_patch * self.p * dim_in # 2 *3 *2 -> 24 in each patch
        bound = 1.0 / math.sqrt(fan_in)
        self.w = nn.Parameter(torch.empty(self.G, fan_in, dim).uniform_(-bound, bound))  # 4, 24, dim
        self.b = nn.Parameter(torch.zeros(self.G, dim))  # 4, dim
        self.norm = nn.LayerNorm(dim)
        self.group_embed = nn.Parameter(torch.randn(self.G, dim) * 0.02)
        self.spatial = nn.ModuleList(SpatialBlock(dim, heads) for _ in range(n_blocks))
        self.temporal = nn.ModuleList(Block(dim, heads, window, 2 ** i) for i in range(n_blocks))
        self.merge = nn.Linear(self.G * dim, dim)
        self.merge_norm = nn.LayerNorm(dim)

 
    def tokenize(self, x):
        """
        (B, T, V, C) -> (B, T/t, G, D), before the norm. Linear in each patch.
        """
        B, T, V, C = x.shape
        assert V == self.num_joints and C == self.dim_in, f"expected (*, *, {self.num_joints}, {self.dim_in}), got {tuple(x.shape)}"
        assert T % self.t_patch == 0, f"T={T} must be divisible by t_patch={self.t_patch}"
        
        x = x[:, :, self.idx].reshape(B, T // self.t_patch, self.t_patch, self.G, self.p, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, T // self.t_patch, self.G, -1)
        # unlike conv, the weights are not shared across joint groups.
        return torch.einsum("btgi,gid->btgd", x, self.w) + self.b

 
    def forward(self, x, mask=None, mask_token=None):
        """
        mask: (B, T/t_patch) bool, True = replace that time step's tokens (all groups).
        """
        h = self.norm(self.tokenize(x))                     # (B,T',G,D)
        if mask is not None:
            h = torch.where(mask[:, :, None, None], mask_token.to(h.dtype), h)
        
        h = h + self.group_embed                            # after masking: groups stay distinct
        B, Tt, G, D = h.shape
        for sp, tp in zip(self.spatial, self.temporal):
            h = sp(h.reshape(B * Tt, G, D)).reshape(B, Tt, G, D)
            h = tp(h.permute(0, 2, 1, 3).reshape(B * G, Tt, D))
            h = h.reshape(B, G, Tt, D).permute(0, 2, 1, 3)
        return self.merge_norm(self.merge(h.reshape(B, Tt, G * D)))
 
 
class Down(nn.Module):
    def __init__(self, dim, r):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, r, stride=r)
        self.norm = nn.LayerNorm(dim)
 
    def forward(self, x):
        return self.norm(self.conv(x.transpose(1, 2)).transpose(1, 2))