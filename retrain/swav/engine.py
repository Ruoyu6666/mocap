
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from retrain.swav.layers import ProjectionHead, PrototypeLayer
from retrain.swav.model import SwAVSkeletonModel


def train_one_epoch_clip(model: SwAVSkeletonModel, dataloader, optimizer, device,
                         max_shift: int = 2, min_sep: int = 1,
                         freeze_prototypes_epoch: bool = False,
                         log_every: int = 50,):
    """
    Use this instead of train_one_epoch when your encoder consumes a whole clip at once and returns per-frame embeddings 
    for every frame in it (encoder(clip) -> (B, T, D)), rather than one embedding per single-frame input.
 
    Each batch is (clip, center_idx):
      clip:       (B, T, J, C) — encoder processes this ONCE per batch
      center_idx: (B,) anchor frame index within each clip
 
    The two SwAV views are two time-indices' embeddings pulled from that single forward pass — no second encoder call needed, 
    and each frame's embedding still has full transformer context from the whole clip.
 
    Batch composition note: shuffling here (via a shuffled DataLoader) only changes which CLIPS appear together in a batch and 
    in what order across epochs. It does NOT reorder frames within a clip — the encoder always sees each clip's frames in their 
    original temporal order. If your clips are a dense/overlapping sliding window over long sequences, prefer a larger window stride 
    (or a sequence-aware sampler) so a batch isn't dominated by near-duplicate overlapping clips.
    """
    model.train()
    running_loss = 0.0
 
    for step, (clip, center_idx) in enumerate(dataloader):
        clip = clip.to(device)               # (B, T, J, C)
        center_idx = center_idx.to(device)   # (B,)
 
        if model.mode == "freeze":
            with torch.no_grad():
                z_seq = model.encoder(clip)  # (B, T, D)
        else:
            z_seq = model.encoder(clip)
 
        T = z_seq.shape[1]
        idx_a, idx_b = sample_two_time_indices(center_idx, T, max_shift=max_shift, min_sep=min_sep)
 
        arange_b = torch.arange(z_seq.shape[0], device=device)
        z_a = z_seq[arange_b, idx_a]  # (B, D)
        z_b = z_seq[arange_b, idx_b]  # (B, D)
 
        p_a = model.projection_head(z_a) if model.projection_head is not None else z_a
        p_b = model.projection_head(z_b) if model.projection_head is not None else z_b
        p_a = F.normalize(p_a, dim=1, p=2)
        p_b = F.normalize(p_b, dim=1, p=2)
 
        scores_a = model.prototypes(p_a)
        scores_b = model.prototypes(p_b)
 
        loss = swav_loss(scores_a, scores_b)
 
        optimizer.zero_grad()
        loss.backward()
 
        if freeze_prototypes_epoch:
            for p in model.prototypes.parameters():
                p.grad = None
 
        optimizer.step()
        model.prototypes.normalize_prototypes()
 
        running_loss += loss.item()
        if step % log_every == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
 
    return running_loss / max(1, len(dataloader))
 



# --------------------------------------------------------------------------
# 6. Example training loop (single-frame-input encoder variant)
# --------------------------------------------------------------------------
def train_one_epoch(model: SwAVSkeletonModel,
                    dataloader,
                    optimizer,
                    device: str = "cuda",
                    max_shift: int = 2,
                    min_sep: int = 1,
                    freeze_prototypes_epoch: bool = False,
                    log_every: int = 50,):
    """
    Expects each batch to be (sequence, center_idx):
      sequence:   (B, T, J, C) — a small temporal window around each anchor frame, T >= 2*max_shift + 1
      center_idx: (B,) index of the anchor frame within each window (usually
                  a constant, e.g. T // 2, unless your windows are ragged
                  near sequence boundaries)
 
    No SkeletonAugment is applied — the two SwAV views are two distinct frames sampled from the window via sample_two_temporal_views.
    """
    model.train()
    running_loss = 0.0
 
    for step, (sequence, center_idx) in enumerate(dataloader):
        sequence = sequence.to(device)
        center_idx = center_idx.to(device)
 
        x_a, x_b = sample_two_temporal_views(sequence, center_idx, max_shift=max_shift, min_sep=min_sep)
 
        z_a, p_a, scores_a = model(x_a)
        z_b, p_b, scores_b = model(x_b)
 
        loss = swav_loss(scores_a, scores_b)
 
        optimizer.zero_grad()
        loss.backward()
 
        if freeze_prototypes_epoch:
            # standard SwAV trick: freeze prototype gradients for epoch 0
            for p in model.prototypes.parameters():
                p.grad = None
 
        optimizer.step()
        model.prototypes.normalize_prototypes()  # keep C on unit sphere
 
        running_loss += loss.item()
        if step % log_every == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
 
    return running_loss / max(1, len(dataloader))
 