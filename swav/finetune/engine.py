import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from datasets.augmentations import Augmentations, _resample_time
from swav.finetune.model import SwAVSkeletonModel
from swav.finetune.utils import swav_loss, sample_binned_time_indices, swav_loss_multiview, align_labels_to_tokens, pool_sequence

import numpy as np


def train_one_epoch_clip_augmented(model: SwAVSkeletonModel, dataloader, optimizer, augment: Augmentations, 
                                   device, freeze_prototypes_epoch: bool = False, log_every: int = 50,):
    """
    SwAV training where the two views are two independently-augmented copies of the clip, 
    each run through the encoder separately. Every frame position is used as a positive pair between the two views 
    (view_a vs view_b's embedding at frame t, for every t),  flattened into an effective batch of size B*T for the loss.
    """
    model.train()
    running_loss = 0.0
    downsample_rates = (1, 4, 10)
    print(downsample_rates)
    n_views = len(downsample_rates)
    for step, batch in enumerate(dataloader):
        clip = batch[0] if isinstance(batch, (tuple, list)) else batch
        clip = clip.to(device)      # (B, T, J, C)
        B, T, _, _ = clip.shape
        scores_list = []
        for rate in downsample_rates:
            if rate == 1:
                clip_v_np = clip.clone().cpu().numpy()
                for i in range(B):
                    clip_v_np[i] = dataloader.dataset.mocap_normalize(clip_v_np[i])[0]
            else:
                clip_v_np = clip.clone().cpu().numpy()[:, ::rate]
                for i in range(B):
                    seq = augment(clip_v_np[i])
                    clip_v_np[i] = dataloader.dataset.mocap_normalize(seq)[0]
            clip_v = torch.tensor(clip_v_np, dtype=torch.float32).to(device)

            if model.mode == "freeze":
                with torch.no_grad():
                    z_seq_v = model.encoder(clip_v) if rate == 1 else model.encoder(clip_v, rate)
            else:
                z_seq_v = model.encoder(clip_v) if rate == 1 else model.encoder(clip_v, rate)

            z_v = z_seq_v.mean(dim=1)  # (B, D) — pool, sequence
            p_v = model.projection_head(z_v) if model.projection_head is not None else z_v
            p_v = F.normalize(p_v, dim=1, p=2)
            scores_list.append(model.prototypes(p_v))

        loss = swav_loss_multiview(scores_list)
        optimizer.zero_grad()
        loss.backward()
        if freeze_prototypes_epoch:
            for p in model.prototypes.parameters():
                p.grad = None

        optimizer.step()
        model.prototypes.normalize_prototypes()
        running_loss += loss.item()
        if step % log_every == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  (effective batch {B}, n_views={n_views})")

    return running_loss / max(1, len(dataloader))




def train_one_epoch_clip_augmented_sequence_semisup(
    model: SwAVSkeletonModel, dataloader, optimizer, augment: Augmentations, device, 
    n_views: int = 2, pool: str = "mean",
    swav_weight: float = 1.0, cls_weight: float = 1.0, ignore_index: int = -100,
    freeze_prototypes_epoch: bool = False, log_every: int = 100,
):
    """
    Joint clip-level SwAV (augmented_sequence scheme) + frame-level classification
        total_loss = swav_weight * swav_loss + cls_weight * cls_loss
 
    The SwAV loss operates on POOLED (B, D) clip-level embeddings, one per augmented view. 
    The classification loss operates on the UNPOOLED (B, T, D) per-frame embeddings from the FIRST view only, 
    before pooling — labels are aligned to that view's T via align_labels_to_tokens. 
    These are two genuinely  different granularities (clip-level clustering vs frame-level classification) computed 
    from the same n_views  encoder forward passes — classification doesn't add any extra encoder compute.
 
    Requires model.classifier_head to be set. Batches must be (clip, labels): labels (B, T_raw), 
    ignore_index (-100 default) marking unlabeled frames.
    """
    assert model.classifier_head is not None, (
        "model.classifier_head must be set (pass a ClassifierHead to SwAVSkeletonModel) for semi-supervised training")
    model.train()
    running_loss, running_swav, running_cls = 0.0, 0.0, 0.0
    mode = getattr(model, "mode", "finetune")

    downsample_rates = (1, 4, 10)
    print(downsample_rates)
    n_views = len(downsample_rates)

    for step, (clip, labels) in enumerate(dataloader):
        clip = clip.to(device)      # (B, T, J, C)
        B, T, _, _ = clip.shape
        labels = labels.to(device)  # (B, T_raw)
        scores_list = []
        z_seq_first = None

        for v in range(n_views):
            clip_v = augment(clip)
            if mode == "freeze":
                with torch.no_grad():
                    z_seq_v = model.encoder(clip_v)  # (B, T, D)
            else:
                z_seq_v = model.encoder(clip_v)
            if v == 0:
                z_seq_first = z_seq_v  # kept unpooled for classification
 
            z_v = pool_sequence(z_seq_v, pool)  # (B, D)
            p_v = model.projection_head(z_v) if model.projection_head is not None else z_v
            p_v = F.normalize(p_v, dim=1, p=2)
            scores_list.append(model.prototypes(p_v))
 
        loss_swav = swav_loss_multiview(scores_list)
 
        # --- supervised frame-level classification (first view, unpooled) ---
        labels_aligned = align_labels_to_tokens(labels, T)  # (B, T)
        logits = model.classifier_head(z_seq_first)         # (B, T, num_classes)
        valid = (labels_aligned != ignore_index)
        if valid.any():
            loss_cls = F.cross_entropy(
                logits.reshape(B * T, -1), labels_aligned.reshape(B * T), ignore_index=ignore_index,
            )
        else:
            loss_cls = torch.zeros((), device=device)
 
        loss = swav_weight * loss_swav + cls_weight * loss_cls
        optimizer.zero_grad()
        loss.backward()
        if freeze_prototypes_epoch:
            for p in model.prototypes.parameters():
                p.grad = None
 
        optimizer.step()
        model.prototypes.normalize_prototypes()
        running_loss += loss.item()
        running_swav += loss_swav.item()
        running_cls += loss_cls.item()
        if step % log_every == 0:
            print(f"step {step:5d}  total {loss.item():.4f}  swav {loss_swav.item():.4f}  cls {loss_cls.item():.4f}  "
                  f"(labeled frames: {valid.sum().item()}/{B*T})")
 
    n = max(1, len(dataloader))
    return running_loss / n, running_swav / n, running_cls / n






def train_one_epoch_clip_augmented_semisup(
    model: SwAVSkeletonModel,
    dataloader,
    optimizer,
    augment: Augmentations,
    device: str = "cuda",
    swav_weight: float = 1.0,
    cls_weight: float = 1.0,
    ignore_index: int = -100,
    freeze_prototypes_epoch: bool = False,
    log_every: int = 50,
):
    """
    Joint SwAV (frame-level, augmented-pair scheme — see) + frame-level classification loss:
 
        total_loss = swav_weight * swav_loss + cls_weight * cls_loss
 
    Requires model.classifier_head to be set. Batches must be
    (clip, labels): labels (B, T_raw) integer class indices, ignore_index
    (default -100) marking unlabeled frames — supports partial/sparse
    labeling, same as train_one_epoch_clip_binned_semisup.
 
    The classification loss uses view_a's (the first augmented copy's) z_seq only — labels are aligned to 
    its T via align_labels_to_tokens. Only one view is classified, not both, to avoid doubling the
    classification compute; if you want the classifier trained on both augmented views too (a mild 
    consistency-regularization effect), average the two views' cls losses instead.
    """
    assert model.classifier_head is not None, (
        "model.classifier_head must be set (pass a ClassifierHead instance "
        "to SwAVSkeletonModel) for semi-supervised training"
    )
    model.train()
    running_loss, running_swav, running_cls = 0.0, 0.0, 0.0
 
    for step, (clip, labels) in enumerate(dataloader):
        clip = clip.to(device)      # (B, T, J, C)
        labels = labels.to(device)  # (B, T_raw)
 
        clip_a = augment(clip)
        clip_b = augment(clip)
 
        if model.mode == "freeze":
            with torch.no_grad():
                z_seq_a = model.encoder(clip_a)  # (B, T, D)
                z_seq_b = model.encoder(clip_b)  # (B, T, D)
        else:
            z_seq_a = model.encoder(clip_a)
            z_seq_b = model.encoder(clip_b)
 
        B, T, D = z_seq_a.shape
        z_a = z_seq_a.reshape(B * T, D)
        z_b = z_seq_b.reshape(B * T, D)
 
        p_a = model.projection_head(z_a) if model.projection_head is not None else z_a
        p_b = model.projection_head(z_b) if model.projection_head is not None else z_b
        p_a = F.normalize(p_a, dim=1, p=2)
        p_b = F.normalize(p_b, dim=1, p=2)
 
        scores_a = model.prototypes(p_a)
        scores_b = model.prototypes(p_b)
        loss_swav = swav_loss(scores_a, scores_b)
 
        # --- supervised frame-level classification (view_a only) ---
        labels_aligned = align_labels_to_tokens(labels, T)  # (B, T)
        logits = model.classifier_head(z_seq_a)             # (B, T, num_classes)
        valid = (labels_aligned != ignore_index)
        if valid.any():
            loss_cls = F.cross_entropy(logits.reshape(B * T, -1), labels_aligned.reshape(B * T),
                                    ignore_index=ignore_index,)
        else:
            loss_cls = torch.zeros((), device=device)
        loss = swav_weight * loss_swav + cls_weight * loss_cls
 
        optimizer.zero_grad()
        loss.backward()
        if freeze_prototypes_epoch:
            for p in model.prototypes.parameters():
                p.grad = None
 
        optimizer.step()
        model.prototypes.normalize_prototypes()
        running_loss += loss.item()
        running_swav += loss_swav.item()
        running_cls += loss_cls.item()
        if step % log_every == 0:
            print(f"step {step:5d}  total {loss.item():.4f}  "
                  f"swav {loss_swav.item():.4f}  cls {loss_cls.item():.4f}  "
                  f"(labeled frames: {valid.sum().item()}/{B*T})")
 
    n = max(1, len(dataloader))
    return running_loss / n, running_swav / n, running_cls / n


"""
def train_one_epoch_clip_binned(model: SwAVSkeletonModel, dataloader, optimizer, device,
                                n_bins: int = 5, min_sep: int = 4, freeze_prototypes_epoch: bool = False, log_every: int = 100,):
"""
"""
    Variant of train_one_epoch_clip using sample_binned_time_indices: each clip contributes n_bins independent (view_a, view_b) pairs, 
    one per temporal bin, instead of a single pair sampled around one anchor. 
    All n_bins pairs from a batch are flattened into one larger effective batch of size B * n_bins for the SwAV loss — 
    still just ONE encoder forward pass per clip batch.
 
    Dataloader only needs to yield clips; center_idx (if your Dataset still returns one) is ignored here since bins cover the whole clip.
"""
"""
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(dataloader):
        clip = batch[0] if isinstance(batch, (tuple, list)) else batch
        clip = clip.to(device)  # (B, T, J, C)
 
        if model.mode == "freeze":
            with torch.no_grad():
                z_seq = model.encoder(clip)  # (B, T, D)
        else:
            z_seq = model.encoder(clip)
            
        B, T, D = z_seq.shape
        idx, _ = sample_binned_time_indices(B, T, n_bins=n_bins, n_views=2, min_sep=min_sep, device=device)  # (B, n_bins, 2)
        idx_a = idx[:, :, 0].reshape(-1)  # (B*n_bins,)
        idx_b = idx[:, :, 1].reshape(-1)  # (B*n_bins,)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, n_bins).reshape(-1)
 
        z_a = z_seq[batch_idx, idx_a]  # (B*n_bins, D)
        z_b = z_seq[batch_idx, idx_b]  # (B*n_bins, D)
 
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
            print(f"step {step:5d}  loss {loss.item():.4f}  (effective batch {B*n_bins})")
 
    return running_loss / max(1, len(dataloader))
"""

"""
def train_one_epoch_clip(model: SwAVSkeletonModel, dataloader, optimizer, device, max_shift: int = 2, 
                        min_sep: int = 1, freeze_prototypes_epoch: bool = False, log_every: int = 50,):
"""
"""
    Use this when encoder consumes a whole clip at once and returns per-frame embeddings (encoder(clip) -> (B, T, D))
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
""" model.train()
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
""" 



# --------------------------------------------------------------------------
# 6. Example training loop (single-frame-input encoder variant)
# --------------------------------------------------------------------------
#def train_one_epoch(model: SwAVSkeletonModel, dataloader, optimizer, device: str , max_shift: int=2, min_sep: int=1, 
#                    freeze_prototypes_epoch: bool=False, log_every: int=50,):
"""
    Expects each batch to be (sequence, center_idx):
      sequence:   (B, T, J, C) — a small temporal window around each anchor frame, T >= 2*max_shift + 1
      center_idx: (B,) index of the anchor frame within each window (usually a constant, e.g. T // 2, unless 
                your windows are ragged near sequence boundaries)
    No SkeletonAugment is applied — the two SwAV views are two distinct frames sampled from the window via sample_two_temporal_views.
"""
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
    """
 