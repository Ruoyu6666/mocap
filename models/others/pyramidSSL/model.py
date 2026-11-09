import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math

from layers import *

class PyramidEncoder(nn.Module):
    def __init__(self, stem, dim, depths=(2, 2, 2), windows=(16, 32, 0), strides=(3, 2),
                 heads=4, dropout=0.0):
        """
        depths / windows: one entry per level. strides: downsampling factor between
        consecutive levels (len = levels - 1). windows[l] <= 0 -> global attention at level l.
        Default: 900 -> 300 -> 150 frames.
        """
        super().__init__()
        assert len(depths) == len(windows) == len(strides) + 1
        self.L = len(depths)
        self.strides = tuple(strides)
        self.cum = [1]                                      # frames per step at each level
        for r in self.strides:
            self.cum.append(self.cum[-1] * r)
        self.stride = self.cum[-1]                          # total stride (in tokens)
        self.stem = stem
        self.tf = getattr(stem, "t_patch", 1)               # frames per level-0 token
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # dilation 1,2,4,... inside each level; windows[-1]=0 -> global attention at the top
        self.levels = nn.ModuleList(
            nn.ModuleList(Block(dim, heads, windows[l], 2 ** i, dropout=dropout)
                          for i in range(depths[l]))
                                for l in range(self.L))
        self.down = nn.ModuleList(Down(dim, self.strides[l]) for l in range(self.L - 1))
        self.up_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(self.L - 1))
        self.out_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(self.L))

 
    def forward(self, x, mask=None):
        """
        x (B,T,J,C); mask (B, ceil(T/tf)) bool in TOKEN units, True = replace that token
        by the mask token. Returns outs[l]: (B, ceil(ceil(T/tf)/cum[l]), D).
        """
        B, T = x.shape[:2]
        pad = (-T) % (self.tf * self.stride)
        if pad:
            x = torch.cat([x, x[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            if mask is not None:
                mask = F.pad(mask, (0, (T + pad) // self.tf - mask.shape[1]))
        h = self.stem(x, mask, self.mask_token)
        feats = []
        for l in range(self.L):
            for blk in self.levels[l]:
                h = blk(h)
            feats.append(h)
            if l < self.L - 1:
                h = self.down[l](h)
        outs = [None] * self.L
        outs[-1] = self.out_norms[-1](feats[-1])
        for l in range(self.L - 2, -1, -1):                 # top-down merge (FPN style)
            up = outs[l + 1].repeat_interleave(self.strides[l], dim=1)
            outs[l] = self.out_norms[l](feats[l] + self.up_proj[l](up))
        Tt = math.ceil(T / self.tf)
        return [o[:, :math.ceil(Tt / self.cum[l])] for l, o in enumerate(outs)]
 
 
# ----------------------------------------------------------------------------
# Model: student + EMA teacher + heads
# ----------------------------------------------------------------------------
class CrossScaleSSL(nn.Module):
    def __init__(self, encoder, dim, coarse_level=None, proj_dim=128, n_protos=64):
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.coarse_level = encoder.L - 1 if coarse_level is None else coarse_level
        assert self.coarse_level >= 1
        self.S = encoder.cum[self.coarse_level]              # TOKENS per coarse step
        self.tf = encoder.tf                                 # frames per token
        self.dim = dim
 
        def mlp():
            return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, proj_dim))
        self.proj_fine, self.proj_coarse = mlp(), mlp()      # one head per level
        self.prototypes = nn.Linear(proj_dim, n_protos, bias=False)   # SHARED codebook
        self.pred = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
 
    def scores(self, fine, coarse):
        """fine (B,T,D), coarse (B,Tc,D) -> prototype scores (B,T,K), (B,Tc,K)."""
        zf = self.prototypes(F.normalize(self.proj_fine(fine), dim=-1))
        zc = self.prototypes(F.normalize(self.proj_coarse(coarse), dim=-1))
        return zf, zc
 
    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.weight.data = F.normalize(self.prototypes.weight.data, dim=1)
 
    @torch.no_grad()
    def ema_update(self, m):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
 
    @torch.no_grad()
    def embed_full(self, x, concat_coarse=False):
        """Per-frame embeddings for a full long sequence (any T)."""
        self.student.eval()
        T = x.shape[1]
        outs = self.student(x)
        up = lambda z, r: z.repeat_interleave(r, dim=1)[:, :T]     # tokens -> frames
        fine = up(outs[0], self.tf)
        if not concat_coarse:
            return fine
        return torch.cat([fine, up(outs[self.coarse_level], self.tf * self.S)], dim=-1)