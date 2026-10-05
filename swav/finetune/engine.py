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
    downsample_rates = (1, 5)
    n_views = len(downsample_rates)
    for step, batch in enumerate(dataloader):
        clip = batch[0] if isinstance(batch, (tuple, list)) else batch
        clip = clip.to(device)      
        B, T, _, _ = clip.shape     # (B, T, J, C)
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

            # calculate  scores
            # Sol 1: loss sequence level
            z_v = z_seq_v.mean(dim=1)  # (B, D) — pool, sequence
            """
            # Sol 2: loss on frame/patch level
            D = z_seq_v.shape[-1]
            if rate ==1:
                z_v = z_seq_v.reshape(B, T//5, 5, D).mean(dim=2)
                z_v = z_v.reshape(-1 ,D)
            elif rate == 5:
                z_v = z_seq_v.reshape(-1 ,D)
            """
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





def train_one_epoch_clip_augmented_sequence_semisup(model: SwAVSkeletonModel,
    dataloader, optimizer,
    augment,
    device,
    downsample_rates=(1, 4, 10),
    swav_weight: float = 1.0,
    cls_weight: float = 1.0,
    ignore_index: int = -100,
    freeze_prototypes_epoch: bool = False,
    log_every: int = 50,
):
    """
    Joint clip-level SwAV + frame-level classification loss
        total_loss = swav_weight * swav_loss + cls_weight * cls_loss
 
    Each entry in downsample_rates gives one view:
    -> "clean" view: no augmentation, just dataloader.dataset's mocap_normalize, encoder called at 
       its native rate. This view's UNPOOLED z_seq is also what the classification loss attaches to.
 
    All views are pooled (mean over time) and compared via swav_loss_multiview .
    Classification doesn't add any extra encoder compute: it reuses the rate=1 view's forward pass, just before pooling.
 
    downsample_rates MUST include 1 — that's the only view with the clip's unsubsampled time resolution, 
    which the classifier attaches to via align_labels_to_tokens. Requires model.classifier_head to be set. 
    Batches must be (clip, labels): labels (B, T_raw) integer class indices, ignore_index (default -100) 
    marking unlabeled frames — supports partial/sparse labeling.
    """
    assert model.classifier_head is not None, (
        "model.classifier_head must be set (pass a ClassifierHead instance " "to SwAVSkeletonModel) for semi-supervised training"
    )
    model.train()
    running_loss, running_swav, running_cls = 0.0, 0.0, 0.0
    n_views = len(downsample_rates)
 
    for step, (clip, labels) in enumerate(dataloader):
        clip = clip.to(device)      # (B, T, J, C)
        labels = labels.to(device)  # (B, T_raw)
        B, T, _, _ = clip.shape
 
        scores_list = []
        z_seq_clean = None
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
 
            if rate == 1:
                z_seq_clean = z_seq_v  # make a unpooled copy for classification
            z_v = z_seq_v.mean(dim=1)  # (B, D) — pool, sequence
            p_v = model.projection_head(z_v) if model.projection_head is not None else z_v
            p_v = F.normalize(p_v, dim=1, p=2)
            scores_list.append(model.prototypes(p_v))

        loss_swav = swav_loss_multiview(scores_list)
 
        # --- supervised frame-level classification (clean/rate=1 view) ---
        _, T_clean, D = z_seq_clean.shape
        labels_aligned = align_labels_to_tokens(labels, T_clean)  # (B, T_clean)
        labels_aligned  = labels_aligned .long() - 1
        logits = model.classifier_head(z_seq_clean)                # (B, T_clean, num_classes)
        valid = (labels_aligned != ignore_index)
        if valid.any():
            loss_cls = F.cross_entropy(
                logits.reshape(B * T_clean, -1), labels_aligned.reshape(B * T_clean),
                ignore_index=ignore_index,
            )
        else:
            loss_cls = torch.zeros((), device=device)
        loss =  cls_weight * loss_cls + swav_weight * loss_swav
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
            print(f"step {step:5d}  total {loss.item():.4f} swav {loss_swav.item():.4f}  cls {loss_cls.item():.4f}  "
                  f"(labeled frames: {valid.sum().item()}/{B*T_clean}, n_views={n_views})")
 
    n = max(1, len(dataloader))
    return running_loss / n, running_swav / n, running_cls / n



def train_one_epoch_clip_classifier_only(model: SwAVSkeletonModel, dataloader, 
                    optimizer, augment=None, device: str = "cuda",
                    ignore_index: int = -100, log_every: int = 50,):
    """
    Requires model.classifier_head to be set. Batches must be (clip, labels): labels (B, T_raw) 
    integer class indices, ignore_index (default -100) marking unlabeled frames — supports 
    partial/sparse labeling. A batch with zero labeled frames is skipped entirely (no
    optimizer step) rather than backpropagating a meaningless zero loss.
    """
    assert model.classifier_head is not None, (
        "model.classifier_head must be set for classification-only training")
    model.train()
    running_loss = 0.0
    n_batches_used = 0
 
    for step, (clip, labels) in enumerate(dataloader):
        clip = clip.to(device)      # (B, T, J, C)
        labels = labels.to(device)  # (B, T_raw)
        clip_in = augment(clip) if augment is not None else clip
 
        if model.mode == "freeze":
            with torch.no_grad():
                z_seq = model.encoder(clip_in)  # (B, T, D)
        else:
            z_seq = model.encoder(clip_in)
 
        B, T, D = z_seq.shape
        labels_aligned = align_labels_to_tokens(labels, T)  # (B, T)
        labels_aligned  = labels_aligned .long() - 1
        valid = (labels_aligned != ignore_index)
        if not valid.any():
            if step % log_every == 0:
                print(f"step {step:5d}  no labeled frames this batch, skipping")
            continue
 
        logits = model.classifier_head(z_seq)  # (B, T, num_classes)
        loss = F.cross_entropy(logits.reshape(B * T, -1), labels_aligned.reshape(B * T),
                               ignore_index=ignore_index,)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        n_batches_used += 1
        if step % log_every == 0:
            print(f"step {step:5d}  cls_loss {loss.item():.4f} (labeled frames: {valid.sum().item()}/{B*T})")
 
    return running_loss / n_batches_used


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
 