"""
Sparse-view cross-time SSL for skeleton sequences.

Input            : (B, 900, 12, 2)  -> 450 tokens (2 frames each) x 4 joint groups

Span             : 6 time steps (12 frames = 0.4 s at 30 fps) -> 75 spans

Teacher (EMA)    : 1 time step per span (random offset), all 4 groups -> 300 tokens
Student          : n_vis tokens per span, drawn ONLY from the other 5 time steps of the span
                   (+ optional fully hidden segments) -> ~300-375 tokens
Predictor        : light cross-attention decoder; queries = (a) teacher positions (fine),
                   (b) segment centres (global)
Losses           : fine  -> smooth-L1 to the teacher's LayerNorm'd token embedding
                   global-> SwAV: student's query code vs Sinkhorn code of the teacher's scores mean-pooled over a segment of `seg_spans` spans
No fusion of the 4 groups during training; the encoder works on (time, group) tokens with RoPE
over the time index. For probing, concatenate the 4 group outputs per time step.
"""
import copy
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap") 
from models.skeletonMAE.model.layers_rope import RotaryTemporalEmbedding
from swav.finetune.utils import sinkhorn


# ----------------------------------------------------------------------------- augmentation
def spatial_aug(x, rot=True, flip=True, noise=0.01):
    """
    x: (B, T, J, 2). Rotation/reflection about the per-sequence centroid, plus Gaussian noise.
    `noise` is in coordinate units: scale it to your data.
    """
    B = x.shape[0]
    mu = x.mean(dim=(1, 2), keepdim=True)
    y = x - mu
    if flip:
        s = (torch.rand(B, 1, 1, device=x.device) < 0.5).float() * -2 + 1
        y = torch.stack([y[..., 0] * s, y[..., 1]], dim=-1)
    if rot:
        a = torch.rand(B, device=x.device) * 2 * math.pi
        c, s_ = a.cos()[:, None, None], a.sin()[:, None, None]
        y = torch.stack([c * y[..., 0] - s_ * y[..., 1], s_ * y[..., 0] + c * y[..., 1]], dim=-1)
    y = y + noise * torch.randn_like(y)
    return y + mu





class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, n_groups, rope_base=10000.0, rope_ratio=0.5):
        super().__init__()
        self.grp = nn.Embedding(n_groups, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, rope_ratio=rope_ratio) for _ in range(depth))
        self.rope = RotaryTemporalEmbedding(dim=self.blocks[0].attn.rot_dim, base=rope_base)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tok, t, g):
        """tok (B,N,D); t (B,N) time index in token units; g (B,N) group id."""
        x = tok + self.grp(g)
        cos, sin = self.rope(t)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.norm(x)


class Predictor(nn.Module):
    """Queries cross-attend to the student's visible tokens. Queries do not see each other."""

    def __init__(self, dim, pred_dim, depth, heads, n_groups, rope_base=10000.0, rope_ratio=0.5):
        super().__init__()
        Dp = pred_dim
        self.inp = nn.Linear(dim, Dp)
        self.q_fine = nn.Parameter(torch.randn(Dp) * 0.02)
        self.q_glob = nn.Parameter(torch.randn(Dp) * 0.02)
        self.grp = nn.Embedding(n_groups, Dp)
        self.blocks = nn.ModuleList(Block(Dp, heads, rope_ratio=rope_ratio, cross=True) for _ in range(depth))
        self.rope = RotaryTemporalEmbedding(dim=self.blocks[0].attn.rot_dim, base=rope_base)
        self.norm = nn.LayerNorm(Dp)
        self.out = nn.Linear(Dp, dim)

    def forward(self, ctx, t_ctx, g_fine, t_fine, t_glob):
        B, Nf, Ng = ctx.shape[0], t_fine.shape[1], t_glob.shape[1]
        qf = self.q_fine + self.grp(g_fine)                               # (B, Nf, Dp)
        qg = self.q_glob.view(1, 1, -1).expand(B, Ng, -1)                 # (B, Ng, Dp)
        q = torch.cat([qf, qg], dim=1)
        cos, sin = self.rope(torch.cat([t_fine, t_glob], dim=1))
        cos_c, sin_c = self.rope(t_ctx)
        c = self.inp(ctx)
        for b in self.blocks:
            q = b(q, cos, sin, c, cos_c, sin_c)
        q = self.out(self.norm(q))
        return q[:, :Nf], q[:, Nf:]


class SwavHead(nn.Module):
    def __init__(self, dim, proj_hidden, proj_out, n_protos):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, proj_hidden), nn.GELU(),
                                 nn.Linear(proj_hidden, proj_out))
        self.protos = nn.Parameter(torch.randn(n_protos, proj_out))

    def forward(self, x):
        z = F.normalize(self.mlp(x), dim=-1)
        return z @ F.normalize(self.protos, dim=-1).t()     # cosine scores (..., K)


def _ppl(sc):  # prototype-usage perplexity of teacher argmax (pre-Sinkhorn), collapse monitor
    cnt = torch.bincount(sc.argmax(-1), minlength=sc.shape[-1]).float()
    p = cnt / cnt.sum()
    return torch.exp(-(p * (p + 1e-8).log()).sum())

def _wmean(v, w):  # mean of v over entries where w==1
    return (v * w).sum() / w.sum().clamp(min=1)



# full model
class SparseSSL(nn.Module):
    def __init__(self, num_frames=900,num_joints=12, n_groups=4, t_patch_size=2, 
                 span=6, mask_ratio=0.75, seg_spans=5, n_block_seg=3,
                 dim_feat=256, depth=6, heads=8, pred_dim=128, pred_depth=2, pred_heads=4,
                 proto_loc=256, proto_glob=128, proj_hidden=1024, proj_out=128,
                 tau_s=0.1, sk_eps=0.05, sk_iters=3,
                 w_loc=1.0, w_reg=1.0, w_glob=0.5, rope_base=10000.0, rope_ratio=0.5):
        """
        T           : frames per training sequence
        n_groups    : joints 0-2 -> group 0, 3-5 -> group 1, ... (check your joint order)
        patch_t     : frames per token
        span        : time steps (tokens) per span
        n_vis       : visible student tokens per span, out of (span-1)*n_groups  5/20
        seg_spans   : spans per global segment
        n_block_seg : segments per sequence with NO visible student tokens (0 disables)
        """
        super().__init__()
        n_tok = num_frames // t_patch_size
        assert n_tok % span == 0, "num_frames/patch_t must be a multiple of span"
        self.n_span = n_tok // span
        assert self.n_span % seg_spans == 0, "n_span must be a multiple of seg_spans"
        self.n_seg = self.n_span // seg_spans
        assert n_block_seg * seg_spans < self.n_span
        
        self.G, self.D, self.spa = n_groups, dim_feat, span
        self.n_vis = min(round((1.0 - mask_ratio) * span * self.G), (span - 1) * self.G)   # per span
        self.seg_spans, self.n_block_seg = seg_spans, n_block_seg
        self.tau_s, self.sk_eps, self.sk_iters = tau_s, sk_eps, sk_iters
        self.w_loc, self.w_reg, self.w_glob = w_loc, w_reg, w_glob

        self.tok = Tokenizer(num_joints, n_groups, t_patch_size, dim_feat)
        self.enc = Encoder(dim_feat, depth, heads, n_groups, rope_base, rope_ratio)
        self.pred = Predictor(dim_feat, pred_dim, pred_depth, pred_heads, n_groups, rope_base, rope_ratio)
        self.head_loc = SwavHead(dim_feat, proj_hidden, proj_out, proto_loc)
        self.head_glob = SwavHead(dim_feat, proj_hidden, proj_out, proto_glob)

        self.tok_t, self.enc_t, self.head_loc_t, self.head_glob_t = (
            copy.deepcopy(m) for m in (self.tok, self.enc, self.head_loc, self.head_glob))
        for m in (self.tok_t, self.enc_t, self.head_loc_t, self.head_glob_t):
            m.requires_grad_(False)

    # ---- EMA
    @torch.no_grad()
    def update_teacher(self, m):
        pairs = ((self.tok, self.tok_t), (self.enc, self.enc_t),
                 (self.head_loc, self.head_loc_t), (self.head_glob, self.head_glob_t))
        for s, t in pairs:
            for ps, pt in zip(s.parameters(), t.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    # ---- view sampling
    def sample_views(self, B, device):
        """Returns
        t_teacher : (B, S)      teacher time step (token units) in each span
        flat      : (B, Nv)     student visible tokens as indices into the flattened (time*G + group) axis
        """
        G, sp, S = self.G, self.span, self.n_span
        off = torch.randint(0, sp, (B, S), device=device)
        t_teacher = torch.arange(S, device=device)[None] * sp + off

        # student candidates: every (offset, group) cell of the span except the teacher's offset
        score = torch.rand(B, S, sp, G, device=device)
        is_teacher = (torch.arange(sp, device=device) == off[..., None])[..., None]
        score = score.masked_fill(is_teacher, 2.0)
        local = score.view(B, S, sp * G).topk(self.n_vis, dim=-1, largest=False).indices
        flat = torch.arange(S, device=device)[None, :, None] * (sp * G) + local         # t*G + g

        span_vis = torch.ones(B, S, dtype=torch.bool, device=device)
        if self.n_block_seg > 0:
            blocked = torch.rand(B, self.n_seg, device=device).topk(self.n_block_seg, dim=-1).indices
            seg_mask = torch.zeros(B, self.n_seg, dtype=torch.bool, device=device)
            seg_mask.scatter_(1, blocked, True)
            span_mask = seg_mask.repeat_interleave(self.seg_spans, dim=1)            # (B,S)
            keep = (~span_mask).nonzero()[:, 1].view(B, -1)
            flat = torch.gather(flat, 1, keep[..., None].expand(-1, -1, self.n_vis))
            span_vis = ~span_mask

        return t_teacher, flat.reshape(B, -1), span_vis


    # ---- training step
    def forward(self, x):
        G, D, S = self.G, self.D, self.n_span
        B, dev = x.shape[0], x.device
        xs, xt = spatial_aug(x), spatial_aug(x)
        t_teacher, flat, span_vis = self.sample_views(B, dev)
        w = span_vis.float()                                                         # (B,S)

        # ---------------- teacher (no grad): one time step per span, all groups
        with torch.no_grad():
            tt = self.tok_t(xt)
            tt = torch.gather(tt, 1, t_teacher[..., None, None].expand(-1, -1, G, D))
            tt = tt.reshape(B, -1, D)
            t_t = t_teacher.repeat_interleave(G, dim=1)
            g_t = torch.arange(G, device=dev).repeat(S).expand(B, -1)
            zt = self.enc_t(tt, t_t.float(), g_t)                                    # (B,S*G,D)

            tgt = F.layer_norm(zt, (D,))                                             # fine regression target
            zt_span = zt.view(B, S, G, D).mean(2)                                    # (B,S,D)

            sc_l = self.head_loc_t(zt_span).reshape(B * S, -1).float()               # local codes
            q_l = sinkhorn(sc_l, self.sk_eps, self.sk_iters)

            sc_g = self.head_glob_t(zt_span).float()                                 # global codes
            sc_g = sc_g.view(B, self.n_seg, self.seg_spans, -1).mean(2).reshape(B * self.n_seg, -1)
            q_g = sinkhorn(sc_g, self.sk_eps, self.sk_iters)

        # ---------------- student: sparse visible tokens
        ts = self.tok(xs).reshape(B, -1, D)
        ts = torch.gather(ts, 1, flat[..., None].expand(-1, -1, D))
        t_s, g_s = (flat // G).float(), flat % G
        zs = self.enc(ts, t_s, g_s)

        seg_len = self.seg_spans * self.span
        t_glob = (torch.arange(self.n_seg, device=dev) * seg_len + (seg_len - 1) / 2).float().expand(B, -1)
        pf, pg = self.pred(zs, t_s, g_t, t_t.float(), t_glob)    # pf (B,S*G,D), pg (B,n_seg,D)

        # ---------------- losses
        # local terms only on spans that have visible student tokens
        reg = F.smooth_l1_loss(pf.float(), tgt.float(), reduction="none").view(B, S, G, D).mean((2, 3))
        l_reg = _wmean(reg, w)

        pf_span = pf.view(B, S, G, D).mean(2)
        logp_l = F.log_softmax(self.head_loc(pf_span).reshape(B * S, -1).float() / self.tau_s, dim=-1)
        l_loc = _wmean(-(q_l * logp_l).sum(-1).view(B, S), w)

        # global term on every segment (blocked segments are where it matters most)
        logp_g = F.log_softmax(self.head_glob(pg).reshape(B * self.n_seg, -1).float() / self.tau_s, dim=-1)
        l_glob = -(q_g * logp_g).sum(-1).mean()

        loss = self.w_loc * l_loc + self.w_reg * l_reg + self.w_glob * l_glob

        with torch.no_grad():
            logs = {"loc": l_loc.detach(), "reg": l_reg.detach(), "glob": l_glob.detach(),
                    "ppl_loc": _ppl(sc_l), "ppl_glob": _ppl(sc_g)}
        return loss, logs

    # ---- features for linear probing
    @torch.no_grad()
    def encode_dense(self, x, use_teacher=False):
        """All time steps, all groups -> (B, Tt, G, D). Concatenate groups for the probe:
        z.flatten(2) gives (B, Tt, G*D), one vector per 2 frames. Works for any length (RoPE)."""
        G, D = self.G, self.D
        tok, enc = (self.tok_t, self.enc_t) if use_teacher else (self.tok, self.enc)
        t = tok(x)
        B, Tt = t.shape[:2]
        tt = torch.arange(Tt, device=x.device).repeat_interleave(G).expand(B, -1).float()
        gg = torch.arange(G, device=x.device).repeat(Tt).expand(B, -1)
        return enc(t.reshape(B, -1, D), tt, gg).view(B, Tt, G, D)


# ----------------------------------------------------------------------------- training loop sketch
def ema_momentum(step, total, start=0.996, end=1.0):
    return end - (end - start) * (math.cos(math.pi * step / total) + 1) / 2


def train(model, loader, total_steps, lr=1.5e-4, wd=0.05, device="cuda"):
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    step = 0
    while step < total_steps:
        for x in loader:                                                  # x: (B, 900, 12, 2)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss, logs = model(x.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.update_teacher(ema_momentum(step, total_steps))
            if step % 50 == 0:
                print(step, {k: round(float(v), 4) for k, v in logs.items()})
            step += 1
            if step >= total_steps:
                break


