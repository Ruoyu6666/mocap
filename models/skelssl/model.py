import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

STEPS_PER_SPAN = 6  # time steps (tokens of 2 frames) per span
FRAMES_PER_SPAN = 12



# model parts
class SkeleTokenizer(nn.Module):
    """
    Patch = patch_t frames x joints_per_group joints -> one token per joint group (separate linear).
    """

    def __init__(self, dim, n_groups=4, joints_per_group=3, patch_t=2):
        super().__init__()
        self.G, self.J, self.P = n_groups, joints_per_group, patch_t
        in_dim = patch_t * joints_per_group * 2
        self.weight = nn.Parameter(torch.randn(n_groups, in_dim, dim) * in_dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(n_groups, dim))
        self.group_emb = nn.Parameter(torch.randn(n_groups, dim) * 0.02)

    def forward(self, x):  
        # (B, T, G*J, 2) -> (B, T/P, G, D); joints must be ordered group by group
        B, T, J, C = x.shape
        x = x.reshape(B, T // self.P, self.P, self.G, self.J, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, T // self.P, self.G, -1)
        return torch.einsum("btgi,gid->btgd", x, self.weight) + self.bias + self.group_emb


class SpanEncoder(nn.Module):
    """
    Level 1: attention inside each span -> pool to 1 token. 
    Level 2: global attention over spans.
    Top-down merge adds the span context back to every fine token.
    """
    def __init__(self, dim, heads, depth1, depth2, rope_base):
        super().__init__()
        self.l1 = nn.ModuleList([Block(dim, heads, rope_base=rope_base) for _ in range(depth1)])
        self.pool = AttnPool(dim, heads)
        self.l2 = nn.ModuleList([Block(dim, heads, rope_base=rope_base) for _ in range(depth2)])
        self.norm2 = nn.LayerNorm(dim)
        self.merge = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, pos1, pos2):
        # x (B,S,K,D) tokens of S spans; pos1 (B,S,K) time position inside the span; pos2 (B,S) span position
        B, S, K, D = x.shape
        h, p = x.reshape(B * S, K, D), pos1.reshape(B * S, K)
        for b in self.l1:
            h = b(h, p)
        span = self.pool(h).view(B, S, D)
        for b in self.l2:
            span = b(span, pos2)
        span = self.norm2(span)
        fine = self.norm_out(h.view(B, S, K, D) + self.merge(span)[:, :, None])
        return fine, span


class ProjHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- masks / sinkhorn / aug
def block_mask(B, S, n_mask, bmin, bmax, device):
    """Exactly n_mask masked spans per sample, built from contiguous blocks. True = masked."""
    m = torch.zeros(B, S, dtype=torch.bool)
    for b in range(B):
        cnt = 0
        while cnt < n_mask:
            L = min(random.randint(bmin, bmax), n_mask - cnt, S)
            s = random.randint(0, S - L)
            m[b, s:s + L] = True
            cnt = int(m[b].sum())
    return m.to(device)


@torch.no_grad()
def sinkhorn(scores, eps, iters):
    Q = torch.exp((scores.float() - scores.float().max()) / eps).t()  # (K, N)
    K, N = Q.shape
    Q /= Q.sum()
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True)
        Q /= K
        Q /= Q.sum(0, keepdim=True)
        Q /= N
    return (Q * N).t()  # (N, K), rows sum to 1


def spatial_aug(x, rot=True, flip=True, noise=0.0):
    """Rotation, reflection, gaussian noise. Assumes coordinates are already centred / scaled."""
    B = x.size(0)
    if rot:
        th = torch.rand(B, device=x.device) * 2 * math.pi
        c, s = th.cos(), th.sin()
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)
        x = torch.einsum("btjc,bdc->btjd", x, R)
    if flip:
        sign = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0)[:, None, None]
        x = torch.stack((x[..., 0] * sign, x[..., 1]), -1)
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    return x


# ----------------------------------------------------------------------------- full model
class SkeleSSL(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
        D, G = a.dim, a.n_groups
        self.G, self.D = G, D
        self.tok = SkeleTokenizer(D, G, a.joints_per_group, a.patch_t)
        self.enc = SpanEncoder(D, a.heads, a.depth1, a.depth2, a.rope_base)
        self.head = ProjHead(D, a.proj_dim)
        self.protos = nn.Linear(a.proj_dim, a.n_protos, bias=False)
        # light predictor
        self.pred_in = nn.Linear(D, a.pred_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, a.pred_dim) * 0.02)
        self.pred_blocks = nn.ModuleList(
            [Block(a.pred_dim, a.heads, rope_base=a.rope_base) for _ in range(a.pred_depth)])
        self.pred_norm = nn.LayerNorm(a.pred_dim)
        self.pred_out = nn.Linear(a.pred_dim, D)
        self.reg_head = nn.Linear(D, G * D)
        # EMA teacher (tokenizer + encoder + projection head)
        self.t_tok, self.t_enc, self.t_head = (copy.deepcopy(m) for m in (self.tok, self.enc, self.head))
        for m in (self.t_tok, self.t_enc, self.t_head):
            m.requires_grad_(False)
        self.register_buffer("tpos", torch.arange(STEPS_PER_SPAN).repeat_interleave(G).float(), persistent=False)

    @torch.no_grad()
    def update_teacher(self, m):
        for ms, mt in [(self.tok, self.t_tok), (self.enc, self.t_enc), (self.head, self.t_head)]:
            for ps, pt in zip(ms.parameters(), mt.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    def trainable_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def forward(self, xa, xb):
        a, G, D = self.a, self.G, self.D
        B, T = xa.shape[:2]
        S = T // FRAMES_PER_SPAN
        n_mask = max(1, min(S - 1, int(round(S * a.mask_ratio))))
        Sv = S - n_mask
        dev = xa.device
        bi = torch.arange(B, device=dev)[:, None]

        mask = block_mask(B, S, n_mask, min(a.block_min, n_mask), min(a.block_max, n_mask), dev)
        order = mask.float().argsort(dim=1, stable=True)  # visible first (ascending), then masked (ascending)
        vis, msk = order[:, :Sv], order[:, Sv:]
        off = torch.randint(0, STEPS_PER_SPAN, (B, S), device=dev)  # teacher time step in each span

        # ---- student: visible spans only
        tok = self.tok(xa).reshape(B, S, STEPS_PER_SPAN, G, D)
        xv = tok[bi, vis].reshape(B, Sv, STEPS_PER_SPAN * G, D)
        pos2_v = vis.float() * STEPS_PER_SPAN + (STEPS_PER_SPAN - 1) / 2
        _, span_v = self.enc(xv, self.tpos.expand(B, Sv, -1), pos2_v)

        # ---- predictor: visible span tokens + mask tokens placed at the teacher's time step
        z = self.pred_in(span_v)
        q_tok = self.mask_token.expand(B, n_mask, -1)
        pos_q = msk.float() * STEPS_PER_SPAN + off.gather(1, msk).float()
        seq, pos = torch.cat([z, q_tok], 1), torch.cat([pos2_v, pos_q], 1)
        for b in self.pred_blocks:
            seq = b(seq, pos)
        out = self.pred_out(self.pred_norm(seq[:, Sv:]))  # (B, n_mask, D)

        # ---- teacher: one time step per span, no mask, no grad
        with torch.no_grad():
            tt = self.t_tok(xb).reshape(B, S, STEPS_PER_SPAN, G, D)
            t_in = tt[bi, torch.arange(S, device=dev)[None], off]  # (B,S,G,D)
            t_pos1 = off[..., None].float().expand(B, S, G)
            t_pos2 = torch.arange(S, device=dev)[None].float() * STEPS_PER_SPAN + off.float()
            t_fine, t_span = self.t_enc(t_in, t_pos1, t_pos2)
            t_span_m, t_fine_m = t_span[bi, msk], t_fine[bi, msk]  # masked spans only
            W = F.normalize(self.protos.weight.detach(), dim=1)
            t_scores = (F.normalize(self.t_head(t_span_m), dim=-1) @ W.t()).flatten(0, 1).float()
            q = sinkhorn(t_scores, a.sinkhorn_eps, a.sinkhorn_iters)
            target = F.layer_norm(t_fine_m.float(), (D,))

        # ---- losses
        Ws = F.normalize(self.protos.weight, dim=1)
        s_scores = (F.normalize(self.head(out), dim=-1) @ Ws.t()).flatten(0, 1).float()
        l_swav = -(q * F.log_softmax(s_scores / a.student_temp, dim=-1)).sum(-1).mean()
        pred = self.reg_head(out).view(B, n_mask, G, D)
        l_reg = F.smooth_l1_loss(pred.float(), target)
        loss = l_swav + a.lambda_reg * l_reg
        return {"loss": loss, "swav": l_swav.detach(), "reg": l_reg.detach(),
                "t_scores": t_scores, "s_scores": s_scores.detach(), "q": q,
                "t_feat": t_span_m.flatten(0, 1).detach()}

    @torch.no_grad()
    def extract(self, x, fuse="concat", teacher=False):
        """Dense, unmasked features for probing. Returns fine (B, T/2, G*D or D) and span (B, S, D)."""
        tk, en = (self.t_tok, self.t_enc) if teacher else (self.tok, self.enc)
        G, D = self.G, self.D
        T = x.size(1) // FRAMES_PER_SPAN * FRAMES_PER_SPAN
        tok = tk(x[:, :T])
        B, N = tok.shape[:2]
        S = N // STEPS_PER_SPAN
        pos2 = (torch.arange(S, device=x.device).float() * STEPS_PER_SPAN + (STEPS_PER_SPAN - 1) / 2).expand(B, S)
        fine, span = en(tok.reshape(B, S, STEPS_PER_SPAN * G, D), self.tpos.expand(B, S, -1), pos2)
        fine = fine.reshape(B, N, G, D)
        return (fine.flatten(2) if fuse == "concat" else fine.mean(2)), span


# ----------------------------------------------------------------------------- data
class PoseCrops(torch.utils.data.Dataset):
    """Random temporal crops of length T from an (N, L, 12, 2) .npy file; synthetic walk if no file."""
    def __init__(self, path, T, n_synth=256):
        self.T = T
        self.data = np.load(path, mmap_mode="r") if path else None
        self.n = len(self.data) if self.data is not None else n_synth

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.data is None:
            base = torch.randn(1, 12, 2) * 0.5
            return base + torch.randn(self.T, 12, 2).cumsum(0) * 0.02
        seq = self.data[i]
        start = random.randint(0, max(0, len(seq) - self.T))
        return torch.from_numpy(np.ascontiguousarray(seq[start:start + self.T])).float()