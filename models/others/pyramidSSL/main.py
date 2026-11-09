"""
One-stage, decoder-free SSL for long skeleton sequences (v3: group-aware stem).
 
  Stem    : SkeleStem tokenizes (2 frames x 3 joints) patches with one projection per joint group, 
            then 2x [spatial attention across the groups of a time step, temporal window attention per group], 
            then concatenates the groups and projects to D.
  Encoder : hierarchical pyramid (local window attention + dilated conv at every level,
            strided downsampling between levels, top-down merge back to token resolution).
  One crop per step. Student: spatial augmentation A + span mask. Teacher (EMA): spatial augmentation B, no mask. 
                     Same frames on both sides, so no crop-overlap logic.
  Losses   : (1) cross-scale SwAV: teacher's coarse codes <- student's segment-pooled FINE scores, 
                                   and teacher's pooled-fine codes <- student's COARSE scores
                 (shared prototypes, one projection head per level), masked segments only.
             (2) masked-latent: student per-frame embeddings regress the teacher's
                 layer-normalised per-frame embeddings at masked frames.
  No reconstruction decoder, no second temporal crop.
 
Shapes (12 joints, t_patch=2, strides=(3,2), T=900 frames):
  x (B,900,12,C) -> tokenizer (B,450,4,D) -> group blocks -> merge -> (B,450,D)
  level0 (B,450,D) -> level1 (B,150,D) -> level2 (B,75,D)    [bottom-up]
  outs[l]: same shapes after top-down merge. outs[0] is the token-resolution embedding (1 token = 2 frames); 
  embed_full() repeats it to per-frame. outs[-1] (1 step = 12 frames, global attention) 
  feeds the coarse side of the SwAV loss.
  ALL masks and loss segments are in TOKEN units (mask_span=6 tokens = 12 frames).
"""
import copy
import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.getcwd()
from utils import *
from layers import *
from model import PyramidEncoder, CrossScaleSSL

 
 
@dataclass
class Cfg:
    crop_len: Optional[int] = None  # FRAMES. None = whole sequence (trimmed to a multiple of
                                    # tf*S = frames per coarse step); or an int multiple of it
    temp: float = 0.1
    sinkhorn_eps: float = 0.05
    lambda_cross: float = 1.0
    lambda_latent: float = 1.0
    cross_masked_only: bool = True
    mask_ratio: float = 0.4
    mask_span: int = 6             # TOKENS (6 tokens = 12 frames = 1 coarse step)
    freeze_protos_epochs: int = 1
    ema_start: float = 0.996
    grad_clip: float = 3.0
 
 
def train_step(model, x, optimizer, augment, cfg: Cfg, epoch: int, step: int, total_steps: int):
    """
    x: (B,T,J,C) full sequences. augment: SPATIAL only, called twice (student / teacher)
    so each side gets its own random draw. Same frames on both sides.
    """
    model.student.train()
    model.teacher.eval()
    B, T = x.shape[:2]
    unit = model.tf * model.S                               # frames per coarse step
    L = (T // unit) * unit if cfg.crop_len is None else cfg.crop_len
    assert L % unit == 0 and 0 < L <= T, f"crop {L} must be a multiple of {unit} and <= T={T}"
    s0 = random.randint(0, T - L)          # full length: drops < one coarse step at a random end
    crop = x[:, s0:s0 + L]
    v_stu, v_tea = augment(crop), augment(crop)
    mask = block_mask(B, L // model.tf, cfg.mask_span, cfg.mask_ratio, x.device)   # token units
 
    model.normalize_prototypes()
    c = model.coarse_level
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                        enabled=x.device.type == "cuda"):
        with torch.no_grad():
            to = model.teacher(v_tea)                       # clean, EMA weights
            tzf, tzc = model.scores(to[0], to[c])
        so = model.student(v_stu, mask=mask)                # masked student
        szf, szc = model.scores(so[0], so[c])
 
        l_cross, sel = cross_scale_swav(model, szf, szc, tzf, tzc, mask, cfg.temp, cfg.sinkhorn_eps, cfg.cross_masked_only)
        l_lat = masked_latent_loss(model, so[0], to[0], mask)
        loss = cfg.lambda_cross * l_cross + cfg.lambda_latent * l_lat
 
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if epoch < cfg.freeze_protos_epochs:
        for p in model.prototypes.parameters():
            p.grad = None
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
    optimizer.step()
 
    m = 1 - (1 - cfg.ema_start) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2
    model.ema_update(m)
 
    with torch.no_grad():                                   # collapse monitors
        K = tzc.shape[-1]
        pc = F.softmax(tzc.float() / cfg.temp, -1).reshape(-1, K).mean(0)
        pf = F.softmax(seg_mean(tzf.float(), model.S) / cfg.temp, -1).reshape(-1, K).mean(0)
        use = lambda p: torch.exp(-(p * (p + 1e-9).log()).sum()).item()
        fstd = to[0].float().reshape(-1, to[0].shape[-1]).std(0).mean().item()
    return {"loss": loss.item(), "cross": l_cross.item(), "latent": l_lat.item(),
            "usage_coarse": use(pc), "usage_fine": use(pf), "feat_std": fstd,
            "sel_frac": sel, "ema_m": m}
 
 
# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0); random.seed(0)
    V, C, D, K = 12, 2, 64, 32 # _, _, dimension, num_prototypes
    stem = SkeleStem(D, num_joints=V, dim_in=C, patch_size=3, t_patch=2)
    enc = PyramidEncoder(stem, D)                         # 900 frames -> 450 -> 150 -> 75
    model = CrossScaleSSL(enc, D, n_protos=K)
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
 
    # 1) tokenizer == your Conv2d patch embed when the weights are shared
    conv = nn.Conv2d(C, D, kernel_size=[2, 3], stride=[2, 3])
    xt = torch.randn(2, 20, V, C)
    ref = conv(torch.einsum("ntsc->ncts", xt)).permute(0, 2, 3, 1)         # (N, T/2, G, D)
    w = conv.weight.permute(2, 3, 1, 0).reshape(2 * 3 * C, D)              # (t, p, C) -> D
    with torch.no_grad():
        stem.w.copy_(w.expand(stem.G, -1, -1)); stem.b.copy_(conv.bias.expand(stem.G, -1))
        mine = stem.tokenize(xt)
    print("tokenizer matches Conv2d patch embed:", torch.allclose(ref, mine, atol=1e-5))
 
    x = torch.randn(4, 900, V, C)
    print("level shapes T=900:", [tuple(o.shape) for o in enc(x)])
    print("level shapes T=905:", [tuple(o.shape) for o in enc(torch.randn(2, 905, V, C))])
    print("embed_full:", tuple(model.embed_full(x[:, :895]).shape),
          "concat:", tuple(model.embed_full(x[:, :895], True).shape))
 
    # 2) no leakage: changing frames under a masked token cannot change the output
    enc.eval()
    m = block_mask(4, 450, 6, 0.4, x.device)
    x2 = x.clone()
    fm = m.repeat_interleave(2, dim=1)                                     # token mask -> frames
    x2[fm] = torch.randn_like(x2[fm])
    print("masked frames do not leak:", torch.allclose(enc(x, mask=m)[0], enc(x2, mask=m)[0], atol=1e-5))
    enc.train()
 
    cfg = Cfg(freeze_protos_epochs=1)          # crop_len=None -> full 900-frame sequences
    opt = build_optimizer(model)
    aug = lambda v: v + 0.05 * torch.randn_like(v)          # replace with rotation / reflection etc.
    t0 = {n: p.clone() for n, p in model.teacher.named_parameters()}
    total = 10
    for step in range(total):
        out = train_step(model, x, opt, aug, cfg, epoch=0 if step < 2 else 1, step=step, total_steps=total)
        print(step, {k: round(v, 4) for k, v in out.items()})
    moved = max((p - t0[n]).abs().max().item() for n, p in model.teacher.named_parameters())
    print("teacher moved by EMA:", moved > 0)
    print("teacher requires_grad any:", any(p.requires_grad for p in model.teacher.parameters()))
    out = train_step(model, torch.randn(2, 905, V, C), opt, aug, cfg, 1, 0, total)
    print("T=905 ok:", round(out["loss"], 3))
    out = train_step(model, torch.randn(2, 1500, V, C), opt, aug, Cfg(crop_len=600), 1, 0, total)
    print("1500 -> crop 600 ok:", round(out["loss"], 3))