import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.getcwd()
from layers import *

# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
@torch.no_grad()
def sinkhorn(scores, eps=0.05, iters=3):
    scores = scores.float()
    scores = scores - scores.max()
    Q = torch.exp(scores.float() / eps).t()                 # (K,N)
    K, N = Q.shape
    Q /= Q.sum()
    for _ in range(iters):
        Q /= Q.sum(dim=1, keepdim=True); Q /= K
        Q /= Q.sum(dim=0, keepdim=True); Q /= N
    return (Q * N).t()                                      # (N,K) soft codes
 
 
def seg_mean(z, S):
    """
    (B,T,K) -> (B,T/S,K): mean over each coarse segment.
    """
    B, T, K = z.shape
    return z.reshape(B, T // S, S, K).mean(2)
 
 
def cross_scale_swav(model, s_fine, s_coarse, t_fine, t_coarse, mask, temp=0.1,
                     eps=0.05, masked_only=True):
    """
    Student scores (s_*) predict teacher codes across levels.
      fine -> coarse : student's pooled fine scores   predict teacher's coarse codes
      coarse -> fine : student's coarse scores        predict teacher's pooled fine codes
    Returns (loss, fraction_of_segments_used).
    """
    S = model.S
    B, Tc, K = s_coarse.shape
    with torch.no_grad():
        q_c = sinkhorn(t_coarse.float().reshape(-1, K), eps)               # (B*Tc,K)
        q_f = sinkhorn(seg_mean(t_fine.float(), S).reshape(-1, K), eps)
    lp_f = F.log_softmax(seg_mean(s_fine.float(), S).reshape(-1, K) / temp, -1)
    lp_c = F.log_softmax(s_coarse.float().reshape(-1, K) / temp, -1)
    per = -0.5 * ((q_c * lp_f).sum(-1) + (q_f * lp_c).sum(-1))              # (B*Tc,)
    
    if masked_only:                                         # steps with >=1 masked frame
        sel = mask.reshape(B, Tc, S).any(-1).reshape(-1).float()
    else:
        sel = torch.ones_like(per)
    return (per * sel).sum() / sel.sum().clamp(min=1), sel.mean().item()
 
 
def masked_latent_loss(model, student_fine, teacher_fine, mask):
    D = teacher_fine.shape[-1]
    tgt = F.layer_norm(teacher_fine.float(), (D,))          # data2vec-style normalised target
    pred = model.pred(student_fine).float()
    return F.smooth_l1_loss(pred[mask], tgt[mask])
 
 
def block_mask(B, T, span, ratio, device):
    """
    Random span mask: time cut into blocks of `span` frames (random phase), each block
    masked with prob `ratio`, at least one block per sample.
    """
    off = random.randrange(span)
    idx = (torch.arange(T, device=device) + off) // span
    nb = int(idx.max()) + 1
    m = torch.rand(B, nb, device=device) < ratio
    m[torch.arange(B, device=device), torch.randint(0, nb, (B,), device=device)] = True
    return m[:, idx]                                        # (B,T)
 
 
# ----------------------------------------------------------------------------
# Optimizer / step
# ----------------------------------------------------------------------------
def build_optimizer(model, lr=5e-4, wd=0.05):
    params = [p for p in model.parameters() if p.requires_grad]   # teacher is frozen
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)