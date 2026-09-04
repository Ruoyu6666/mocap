import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from swav.finetune.model import SwAVSkeletonModel

# --------------------------------------------------------------------------
# 2. Sinkhorn-Knopp: turns raw scores into equipartitioned soft codes
# --------------------------------------------------------------------------
@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """
    scores:    (B, K) similarity logits (z @ C), NOT yet softmaxed.
    Returns Q: (B, K) soft assignment, each row sums to 1, each column
    (approximately) equal mass -> prevents collapse to a few prototypes.
    """
    Q = torch.exp(scores / eps).T  # (K, B)
    B = Q.shape[1]
    K = Q.shape[0]
    # normalize Q to be a valid transport plan
    sum_Q = Q.sum()
    Q /= sum_Q
 
    for _ in range(n_iters):
        # normalize rows (each prototype gets equal total mass)
        row_sum = Q.sum(dim=1, keepdim=True)
        Q /= row_sum
        Q /= K
        # normalize columns (each sample sums to 1)
        col_sum = Q.sum(dim=0, keepdim=True)
        Q /= col_sum
        Q /= B
 
    Q *= B      # columns sum to 1 (soft assignment per sample)
    return Q.T  # (B, K)



# --------------------------------------------------------------------------
# 4. SwAV loss
# --------------------------------------------------------------------------
def swav_loss(scores_a: torch.Tensor, scores_b: torch.Tensor,
              sinkhorn_eps: float = 0.05, sinkhorn_iters: int = 3,
              temperature: float = 0.1,) -> torch.Tensor:
    """
    scores_a, scores_b: (B, K) raw prototype logits (z @ C) for two views.
    Swapped prediction: codes from view A predict view B's distribution and vice versa.
    """
    with torch.no_grad():
        q_a = sinkhorn(scores_a, eps=sinkhorn_eps, n_iters=sinkhorn_iters)
        q_b = sinkhorn(scores_b, eps=sinkhorn_eps, n_iters=sinkhorn_iters)
 
    p_a = F.log_softmax(scores_a / temperature, dim=1)
    p_b = F.log_softmax(scores_b / temperature, dim=1)
 
    loss = -0.5 * (
        (q_b * p_a).sum(dim=1).mean() + (q_a * p_b).sum(dim=1).mean()
    )
    return loss




def compute_bin_ranges(T: int, n_bins: int):
    """
    Splits [0, T) into n_bins contiguous, roughly-equal ranges e.g. T=50, n_bins=5 -> [(0,10), (10,20), (20,30), (30,40), (40,50)]
    """
    edges = torch.linspace(0, T, n_bins + 1).round().long()
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_bins)]




def sample_binned_time_indices(B: int, T: int, n_bins: int=5, n_views: int=2, min_sep: int = 4, device: str = "cpu",):
    """
    Returns:
      idx: (B, n_bins, n_views) long tensor of absolute time indices
      bin_ranges: list of (start, end) tuples, for reference/debugging
    """
    bin_ranges = compute_bin_ranges(T, n_bins)
    idx = torch.empty(B, n_bins, n_views, dtype=torch.long, device=device)
 
    for bi, (start, end) in enumerate(bin_ranges):
        L = end - start
        candidates = torch.arange(L, device=device)  # local positions within this bin
        n_cand = candidates.numel()
        assert n_views <= n_cand, (f"bin {bi} ([{start},{end})) has only {n_cand} frames, cannot sample "
                                   f"{n_views} views from it — use fewer bins, fewer views, or a longer clip")
 
        chosen = torch.empty(B, n_views, dtype=torch.long, device=device)
        valid = torch.ones(B, n_cand, dtype=torch.bool, device=device)
 
        for v in range(n_views):
            row_sum = valid.float().sum(dim=1, keepdim=True)
            if (row_sum == 0).any():
                raise ValueError(f"bin {bi} ([{start},{end}), length {L}): no frame left to satisfy min_sep={min_sep} for view {v}. "
                                f"Reduce min_sep, n_views, or n_bins, or lengthen the clip.")
            probs = valid.float() / row_sum
            pick = torch.multinomial(probs, 1).squeeze(1)  # (B,) index into candidates
            chosen[:, v] = pick
            pos_v = candidates[pick]
            all_pos = candidates.unsqueeze(0).expand(B, -1)
            too_close = (all_pos - pos_v.unsqueeze(1)).abs() < min_sep
            valid = valid & (~too_close)
 
        idx[:, bi, :] = candidates[chosen] + start
 
    return idx, bin_ranges



# --------------------------------------------------------------------------
# 7. Initializing prototypes from your existing GMM / watershed centers
# --------------------------------------------------------------------------
def init_prototypes_from_gmm(model: SwAVSkeletonModel, gmm_means, embed_normalize=True):
    """
    gmm_means: numpy array or tensor (K, D) — means of your existing GMM fit on pretrained SkeletonMAE embeddings. 
    K must equal model.prototypes' num_prototypes and D must equal embed_dim.
    """
    centers = torch.as_tensor(gmm_means, dtype=torch.float32)
    if embed_normalize:
        centers = F.normalize(centers, dim=1, p=2)
    model.prototypes.init_from_centers(centers)



def build_optimizer(model: SwAVSkeletonModel, lr: float = 1e-4, weight_decay: float = 1e-4):
    """
    Only passes parameters with requires_grad=True to the optimizer.
    Matters most in 'freeze' / 'finetune_last_n' modes — otherwise you'd waste memory on optimizer state 
    (e.g. AdamW momentum buffers) for parameters that never get a gradient.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def compute_new_representations_overlapping(model, dataloader, device: str = "cuda",
                                            which: str = "projection", t_patch_size: int = 1,):
    """
    Differences from your original code, both defensive fixes:
      - no torch.squeeze() before indexing per-item latents — squeeze() would also collapse the batch dimension 
        if the last batch happens to have batch_size 1, silently misindexing subsequent items.
      - count_sum is clamped to a minimum of 1 before dividing, so any token position never covered by a window 
        returns 0 instead of NaN (rather than propagating NaN through everything downstream).
 
    Returns: (num_sequences, full_len // t_patch_size, D_out) tensor, or (..., K) if which='cluster'. 
            Positions never covered by any window are 0 in the output — check the returned count_sum-derived coverage
            yourself if you need to distinguish "0 embedding" from "uncovered".
    """
    assert which in ("projection", "raw", "cluster")
    model.eval()
    dataset = dataloader.dataset
    num_sequences = dataset.num_sequences
    full_len = dataset.seq_keypoints.shape[1]
    T_tokens = full_len // t_patch_size
    count_sum = torch.zeros(num_sequences, T_tokens, 1)
    repr_sum = None  # allocated once we know D_out, from the first batch
    
    item_ptr = 0
    for i, (x, _)  in enumerate(tqdm(dataloader)):
        x = x.to(device)
        B = x.shape[0]
        z_seq = model.encoder(x)  # (B, latent_len, D)
        latent_len = z_seq.shape[1]
 
        if which == "raw":
            out = z_seq
        else:
            flat = z_seq.reshape(-1, z_seq.shape[-1])
            p = model.projection_head(flat) if model.projection_head is not None else flat
            p = F.normalize(p, dim=1, p=2)

            if which == "projection":
                out = p.reshape(B, latent_len, -1)
            else:  # cluster
                scores = model.prototypes(p)
                probs = F.softmax(scores / 0.1, dim=1)
                out = probs.reshape(B, latent_len, -1)
        out = out.cpu()

        if repr_sum is None:
            D_out = out.shape[-1]
            repr_sum = torch.zeros(num_sequences, T_tokens, D_out)
        keypoints_id_batch = dataset.keypoints_ids[item_ptr:item_ptr + B]
        item_ptr += B
        for j, (seq_id, start_idx) in enumerate(keypoints_id_batch):
            start_token = int(start_idx / t_patch_size)
            end_token = start_token + latent_len
            repr_sum[seq_id, start_token:end_token] += out[j]
            count_sum[seq_id, start_token:end_token] += 1
 
    all_representations = repr_sum / count_sum.clamp(min=1)
    return all_representations














@torch.no_grad()
def compute_new_representations_clip(model: SwAVSkeletonModel, dataloader, device, which: str = "projection",
                                    frame_selection: str = "all", return_indices: bool = False,):
    """
    which:
      "projection" - use model.projection_head(z) if a head exists, else raw z. This is the representation
                    that changed during training if the encoder was frozen and only a head was trained.
      "raw"        - always return the raw encoder output z, regardless of whether a projection head exists 
                    (only meaningfully "new" if the encoder itself was fine-tuned).
      "cluster"    - return the (N, K) softmax-over-prototypes assignment instead of a D-dim embedding.
 
    frame_selection:
      "all"    - return every frame in every clip (REQUIRED if the model was trained with the binned pairing scheme).
      "center" - one row per clip, taken at that clip's center_idx. Only valid if the dataloader yields (clip, center_idx) batches 
 
    return_indices: if True, also returns (clip_ids, frame_ids), each an (N,) long tensor giving which original clip / frame 
    each output row came from — useful for stitching results back into UMAP/behavioral-map analyses grouped by clip or by absolute frame position.
 
    dataloader must be UNSHUFFLED for clip_ids to correctly correspond to original dataset order.
 
    Returns: representation tensor (N, D_out) or (N, K), or a tuple (representation, clip_ids, frame_ids) if return_indices=True.
    """
    assert which in ("projection", "raw", "cluster")
    assert frame_selection in ("all", "center")
    model.eval()
    out = []
    clip_id_out, frame_id_out = [], []
    clip_counter = 0
 
    for i, batch in enumerate(tqdm(dataloader)):
        if isinstance(batch, (tuple, list)):
            clip, center_idx = batch[0], (batch[1] if len(batch) > 1 else None)
        else:
            clip, center_idx = batch, None
        clip = clip.to(device)
        z_seq = model.encoder(clip)  # (B, T, D)
        B, T, D = z_seq.shape
 
        if frame_selection == "all":
            z = z_seq.reshape(B * T, D)
            if return_indices:
                cid = torch.arange(clip_counter, clip_counter + B, device=device)
                cid = cid.unsqueeze(1).expand(B, T).reshape(-1)
                fid = torch.arange(T, device=device).unsqueeze(0).expand(B, T).reshape(-1)
                clip_id_out.append(cid.cpu())
                frame_id_out.append(fid.cpu())
        else:  # "center"
            assert center_idx is not None, ("frame_selection='center' requires the dataloader to yield (clip, center_idx) batches — "
                                            "use frame_selection='all' for models trained with the binned pairing scheme")
            center_idx = center_idx.to(device)
            arange_b = torch.arange(B, device=device)
            z = z_seq[arange_b, center_idx]  # (B, D)
            if return_indices:
                clip_id_out.append(torch.arange(clip_counter, clip_counter + B))
                frame_id_out.append(center_idx.cpu())
 
        clip_counter += B
        if which == "raw":
            out.append(z.cpu())
            continue
        p = model.projection_head(z) if model.projection_head is not None else z
        p = F.normalize(p, dim=1, p=2)
        if which == "projection":
            out.append(p.cpu())
        else:  # cluster
            scores = model.prototypes(p)
            probs = F.softmax(scores / 0.1, dim=1)
            out.append(probs.cpu())
 
    result = torch.cat(out, dim=0)
    if return_indices:
        return result, torch.cat(clip_id_out), torch.cat(frame_id_out)

    return result






"""
def sample_two_temporal_views(sequence: torch.Tensor, center_idx: torch.Tensor, max_shift: int = 2, min_sep: int = 1,):
"""    
"""    
    No hand-crafted augmentation: both SwAV views are two different frames drawn from a small temporal window around an anchor frame. 
    This is the "temporal_shift_view, no augment" setup — positives are definedpurely by temporal proximity.
    sequence: (B, T, J, C) — a window of frames per sample (T should be>= 2*max_shift + 1, 
               centered so center_idx +/- max_shift is in range for most samples)
    Returns: (view_a, view_b), each (B, J, C)
"""
"""    
    B, T, J, C = sequence.shape
    device = sequence.device
    assert 2 * max_shift + 1 >= min_sep + 1, ("max_shift too small for the requested min_sep — widen the window or lower min_sep")
 
    offsets = torch.arange(-max_shift, max_shift + 1, device=device)  # (2*max_shift+1,)
    n_off = offsets.numel()
 
    # sample offset_a freely, then sample offset_b conditioned on being at
    # least min_sep away from offset_a (rejection-free via masked sampling)
    idx_a = torch.randint(0, n_off, (B,), device=device)
    offset_a = offsets[idx_a]
 
    # build a (B, n_off) validity mask: True where |offset - offset_a| >= min_sep
    all_offsets = offsets.unsqueeze(0).expand(B, -1)  # (B, n_off)
    valid = (all_offsets - offset_a.unsqueeze(1)).abs() >= min_sep  # (B, n_off)
 
    # sample uniformly among valid choices per row
    probs = valid.float()
    probs = probs / probs.sum(dim=1, keepdim=True)
    idx_b = torch.multinomial(probs, 1).squeeze(1)  # (B,)
    offset_b = offsets[idx_b]
 
    idx_frame_a = (center_idx + offset_a).clamp(0, T - 1)
    idx_frame_b = (center_idx + offset_b).clamp(0, T - 1)
 
    arange_b = torch.arange(B, device=device)
    view_a = sequence[arange_b, idx_frame_a]  # (B, J, C)
    view_b = sequence[arange_b, idx_frame_b]  # (B, J, C)
 
    return view_a, view_b
"""



"""
def sample_two_time_indices(center_idx: torch.Tensor, T: int, max_shift: int = 2, min_sep: int = 1):
    #Same sampling logic as sample_two_temporal_views, but returns clamped time indices only (no gather) 
    # —  used when you already have per-frame embeddings for a whole clip (B, T, D) from a single encoder 
    # forward pass and just need to pick two time steps out of it.

    device = center_idx.device
    B = center_idx.shape[0]
    assert 2 * max_shift + 1 >= min_sep + 1, (
        "max_shift too small for the requested min_sep — widen the window or lower min_sep")
 
    offsets = torch.arange(-max_shift, max_shift + 1, device=device)
    n_off = offsets.numel()
 
    idx_a = torch.randint(0, n_off, (B,), device=device)
    offset_a = offsets[idx_a]
 
    all_offsets = offsets.unsqueeze(0).expand(B, -1)
    valid = (all_offsets - offset_a.unsqueeze(1)).abs() >= min_sep
    probs = valid.float()
    probs = probs / probs.sum(dim=1, keepdim=True)
    idx_b = torch.multinomial(probs, 1).squeeze(1)
    offset_b = offsets[idx_b]
 
    idx_time_a = (center_idx + offset_a).clamp(0, T - 1)
    idx_time_b = (center_idx + offset_b).clamp(0, T - 1)
    
    return idx_time_a, idx_time_b
"""


#def sample_multiple_time_indices(center_idx: torch.Tensor, T: int, n_views: int = 2, max_shift: int = 4, min_sep: int = 5,):
""" Generalization of sample_two_time_indices to n_views >= 2. Greedily picks offsets one at a time; 
    each new offset must be >= min_sep away from every offset already chosen for that sample,  
    
    center_idx: (B,) index of the anchor frame within each window
    max_shift: maximum frame offset from the anchor for either view
    min_sep: minimum |offset_a - offset_b| enforced between the two sampled offsets, to avoid the degenerate case (near-zero gradient, 
            wasted step). If your frame rate is  high / motion is slow, consider raising it so the two views are actually visually distinct.
    
    Returns idx_time: (B, n_views) long tensor of clamped time indices.
"""
"""
    device = center_idx.device
    B = center_idx.shape[0]
    offsets = torch.arange(-max_shift, max_shift + 1, device=device)
    n_off = offsets.numel() # the total number of elements
    assert n_views <= n_off, (f"n_views={n_views} exceeds available offsets ({n_off}) in the window "
                              f"— increase max_shift or reduce n_views")
 
    chosen_idx = torch.empty(B, n_views, dtype=torch.long, device=device)
    valid = torch.ones(B, n_off, dtype=torch.bool, device=device)
 
    for v in range(n_views):
        probs = valid.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        pick = torch.multinomial(probs, 1).squeeze(1)  # (B,) index into offsets
        chosen_idx[:, v] = pick
        offset_v = offsets[pick]
        all_off = offsets.unsqueeze(0).expand(B, -1)
        too_close = (all_off - offset_v.unsqueeze(1)).abs() < min_sep
        valid = valid & (~too_close)
 
    chosen_offsets = offsets[chosen_idx]  # (B, n_views)
    idx_time = (center_idx.unsqueeze(1) + chosen_offsets).clamp(0, T - 1)
    return idx_time
"""


