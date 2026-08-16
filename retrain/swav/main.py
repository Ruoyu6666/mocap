"""
SwAV-style self-supervised fine-tuning for skeleton frame embeddings.
 
- Pretrained SkeletonMAE encoder maps a whole clip (B, T, J, C) -> (B, T, D): one embedding per frame from a single forward pass.
Use SwAVSkeletonModel.forward_clip/ embed_clip and train_one_epoch_clip / compute_new_representations_clip
for this case — that's the correct path for this shape.

- Prepare a dataloader that yields (clip, center_idx) batches: clip is (B, T, J, C) with frames in original temporal order, center_idx (B,) is
the anchor frame index within each clip.
 
- There is also a legacy single-frame path — SwAVSkeletonModel.forward, train_one_epoch, sample_two_temporal_views — 
for encoders that map a single frame (B, J, C) -> (B, D) directly, i.e. do NOT take a whole clip as input. 
Do not mix the two: pick whichever matches your actual encoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from retrain.swav.layers import ProjectionHead, PrototypeLayer
from retrain.swav.model import SwAVSkeletonModel
from retrain.swav.engine import train_one_epoch, train_one_epoch_clip
from retrain.swav.utils import *

 


 
@torch.no_grad()
def compute_new_representations_clip(model: SwAVSkeletonModel, dataloader, device, which: str = "projection",):
    """
    Run the trained model over all clips and collect a new per-frame
    representation for every frame at each batch's center_idx.
 
    which:
      "projection" - use model.projection_head(z) if a head was supplied, else falls back to raw z. This is the representation
                      that actually changed during training if the encoder was frozen and only a head was trained.
      "raw"        - always return the raw encoder output z, regardless of whether a projection head exists 
                    (only meaningfully "new" if the encoder itself was fine-tuned).
      "cluster"    - return the (N, K) softmax-over-prototypes assignment instead of a D-dim embedding.
 
    Returns a single (N, D_out) or (N, K) tensor, concatenated across the whole dataloader in iteration order.
    """
    assert which in ("projection", "raw", "cluster")
    model.eval()
    out = []
    for clip, center_idx in dataloader:
        clip = clip.to(device)
        center_idx = center_idx.to(device)
        z_seq = model.encoder(clip)  # (B, T, D)
        arange_b = torch.arange(z_seq.shape[0], device=device)
        z = z_seq[arange_b, center_idx]  # (B, D) — the anchor frame's embedding
 
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
 
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------
# Usage sketch — clip-input encoder (encoder(clip) -> (B, T, D))
# --------------------------------------------------------------------------

encoder = load_pretrained_skeletonmae_encoder()   # takes (B, T, J, C) -> (B, T, D)
embed_dim = 128                                   # match your MAE output dim
num_prototypes = 60                               # e.g. match your GMM component count
 
# optional projection head — recommended especially for mode="freeze", since it's the only thing that will actually 
# change the representation if the encoder itself never gets gradients. out_dim can differ from embed_dim.
head = ProjectionHead(in_dim=embed_dim, hidden_dim=256, out_dim=128)
# head = None   # <- use this instead to skip the head and cluster on raw z
 
# --- choose an encoder mode ---
# (a) full fine-tuning:            mode="finetune"
# (b) frozen encoder + head only:  mode="freeze"
# (c) unfreeze last N blocks:      mode="finetune_last_n", unfreeze_n=2,
#                                  encoder_blocks_attr="blocks" <- match your encoder's actual attribute name for its transformer block list
 
model = SwAVSkeletonModel(encoder, embed_dim, num_prototypes, mode="freeze", projection_head=head,).to(device)
 
# optional: warm-start prototypes from your existing GMM clustering (means must be in the same space the prototypes 
# operate on — i.e. in head-output space if you're using a head, not raw embed_dim space)
# init_prototypes_from_gmm(model, gmm.means_)
 
optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-4)
 
# dataloader yields (clip, center_idx):
#   clip:       (B, T, J, C) — a full clip, frames in original temporal order
#   center_idx: (B,) index of the anchor frame within each clip (often a constant like T // 2, unless clips are ragged near sequence
#               boundaries)
# Shuffling this dataloader only changes which clips share a batch — frame order *within* each clip is never touched, 
# so temporal context the encoder sees is unaffected.
 
max_shift = 2
min_sep = 1
 
for epoch in range(num_epochs):
    freeze = (epoch == 0)  # SwAV convention: freeze prototype grads epoch 0
    avg_loss = train_one_epoch_clip(model, dataloader, optimizer, device=device, 
                                    max_shift=max_shift, min_sep=min_sep, freeze_prototypes_epoch=freeze)
    print(f"epoch {epoch}: avg loss {avg_loss:.4f}")
 
# --- after training: extract the new representation ---
# Use an unshuffled dataloader over ALL clips/frames you want represented (e.g. center_idx = a fixed frame per clip, or iterate every valid anchor).
new_repr = compute_new_representations_clip(model, full_dataloader, device=device,
                                             which="projection")  # or "cluster" / "raw"
# new_repr: (N, head.out_dim) if which="projection", (N, K) if "cluster",
# (N, embed_dim) if "raw" (only meaningfully new if encoder was fine-tuned)
 
# NOTE on mode="freeze" without a projection head: since neither the encoder nor any head gets gradients, 
# "raw" representations are literally unchanged from your original MAE embeddings — only the prototype/cluster assignment
# is new. Add a projection_head if you want an actual new D-dim embedding.
