import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap/models") # Adds the current directory to the Python path


"""
SwAV-style self-supervised fine-tuning for skeleton frame embeddings

Assumes you already have:
  - a pretrained SkeletonMAE encoder that maps (B, T, J, C) or (B, J, C) skel 
    input to (B, D) frame-level embeddings
  - a dataloader that yields batches of raw skeleton frames (+ optionally e
    indices / sequence ids, useful for temporal-shift augmen
  tation  or
    avoiding same-sequence batches)

Usage sketch at the bottom of this file.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Prototype layer
# --------------------------------------------------------------------------

class PrototypeLayer(nn.Module):
    """
    Learnable prototypes C in R^{D x K}, kept L2-normalized on the unit sphere
    (per SwAV). Call `normalize_prototypes()` after every optimizer.step().
    """

    def __init__(self, embed_dim: int, num_prototypes: int = 60):
        super().__init__()
        self.prototypes = nn.Linear(embed_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    @torch.no_grad()
    def init_from_centers(self, centers: torch.Tensor):
        """
        centers: (K, D) tensor, e.g. GMM means or watershed-region centroids
        computed on your existing pretrained embeddings. Gives the prototypes
        a behaviorally meaningful head start instead of random init.
        """
        assert centers.shape == self.prototypes.weight.shape, (
            f"expected {self.prototypes.weight.shape}, got {centers.shape}"
        )
        centers = F.normalize(centers, dim=1, p=2)
        self.prototypes.weight.copy_(centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z assumed already L2-normalized
        return self.prototypes(z)


# --------------------------------------------------------------------------
# 2. Sinkhorn-Knopp: turns raw scores into equipartitioned soft codes
# --------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """
    scores: (B, K) similarity logits (z @ C), NOT yet softmaxed.
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

    Q *= B  # columns sum to 1 (soft assignment per sample)
    return Q.T  # (B, K)


# --------------------------------------------------------------------------
# 3. Skeleton-appropriate augmentations
# --------------------------------------------------------------------------

class SkeletonAugment:
    """
    Motion-plausible augmentations for a single frame's joint coordinates.
    Expects x: (B, J, C) where C=2 or 3 (x,y[,z]).

    Adjust joint_pairs for your skeleton's left/right symmetry if you want
    mirroring; leave mirror_prob=0 to disable.
    """

    def __init__(
        self,
        joint_dropout_p: float = 0.1,
        jitter_std: float = 0.01,
        rot_deg: float = 15.0,
        mirror_prob: float = 0.0,
        joint_pairs=None,
    ):
        self.joint_dropout_p = joint_dropout_p
        self.jitter_std = jitter_std
        self.rot_deg = rot_deg
        self.mirror_prob = mirror_prob
        self.joint_pairs = joint_pairs or []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        B, J, C = x.shape

        # 1. Gaussian jitter on coordinates
        x = x + torch.randn_like(x) * self.jitter_std

        # 2. random joint dropout (zero out — encoder should already be
        #    robust to this from MAE pretraining, so it's a mild augmentation)
        if self.joint_dropout_p > 0:
            mask = (torch.rand(B, J, 1, device=x.device) > self.joint_dropout_p).float()
            x = x * mask

        # 3. small random rotation around the "up" axis, if C == 3
        if C == 3 and self.rot_deg > 0:
            theta = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(self.rot_deg)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            R = torch.zeros(B, 3, 3, device=x.device)
            R[:, 0, 0] = cos_t
            R[:, 0, 2] = sin_t
            R[:, 1, 1] = 1.0
            R[:, 2, 0] = -sin_t
            R[:, 2, 2] = cos_t
            x = torch.bmm(x, R.transpose(1, 2))
        elif C == 2 and self.rot_deg > 0:
            theta = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(self.rot_deg)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            R = torch.zeros(B, 2, 2, device=x.device)
            R[:, 0, 0] = cos_t
            R[:, 0, 1] = -sin_t
            R[:, 1, 0] = sin_t
            R[:, 1, 1] = cos_t
            x = torch.bmm(x, R.transpose(1, 2))

        # 4. left-right mirror
        if self.mirror_prob > 0 and self.joint_pairs:
            do_mirror = torch.rand(B, device=x.device) < self.mirror_prob
            if do_mirror.any():
                x_mirror = x.clone()
                x_mirror[..., 0] *= -1  # flip x coordinate
                for (l, r) in self.joint_pairs:
                    x_mirror[:, [l, r], :] = x_mirror[:, [r, l], :]
                x[do_mirror] = x_mirror[do_mirror]

        return x


def sample_two_temporal_views(
    sequence: torch.Tensor,
    center_idx: torch.Tensor,
    max_shift: int = 2,
    min_sep: int = 1,
):
    """
    No hand-crafted augmentation: both SwAV views are just two different
    frames drawn from a small temporal window around an anchor frame. This
    is the "temporal_shift_view, no augment" setup — positives are defined
    purely by temporal proximity.

    sequence: (B, T, J, C) — a window of frames per sample (T should be
              >= 2*max_shift + 1, centered so center_idx +/- max_shift is
              in range for most samples)
    center_idx: (B,) index of the anchor frame within each window
    max_shift: maximum frame offset from the anchor for either view
    min_sep: minimum |offset_a - offset_b| enforced between the two sampled
             offsets, to avoid the degenerate case where both views land on
             the same frame (near-zero gradient, wasted step). Keep this
             >= 1. If your frame rate is high / motion is slow, consider
             raising it so the two views are actually visually distinct.

    Returns: (view_a, view_b), each (B, J, C)
    """
    B, T, J, C = sequence.shape
    device = sequence.device

    assert 2 * max_shift + 1 >= min_sep + 1, (
        "max_shift too small for the requested min_sep — widen the window "
        "or lower min_sep"
    )

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


# --------------------------------------------------------------------------
# 4. SwAV loss
# --------------------------------------------------------------------------

def swav_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 3,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    scores_a, scores_b: (B, K) raw prototype logits (z @ C) for two views.
    Swapped prediction: codes from view A predict view B's distribution and
    vice versa.
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


# --------------------------------------------------------------------------
# 5. Wrapper module: encoder + prototypes
# --------------------------------------------------------------------------

class SwAVSkeletonModel(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, num_prototypes: int = 60):
        super().__init__()
        self.encoder = encoder  # your pretrained SkeletonMAE encoder
        self.prototypes = PrototypeLayer(embed_dim, num_prototypes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)              # (B, D)
        z = F.normalize(z, dim=1, p=2)
        scores = self.prototypes(z)      # (B, K)
        return z, scores


# --------------------------------------------------------------------------
# 6. Example training loop
# --------------------------------------------------------------------------

def train_one_epoch(
    model: SwAVSkeletonModel,
    dataloader,
    optimizer,
    device: str = "cuda",
    max_shift: int = 2,
    min_sep: int = 1,
    freeze_prototypes_epoch: bool = False,
    log_every: int = 50,
):
    """
    Expects each batch to be (sequence, center_idx):
      sequence:   (B, T, J, C) — a small temporal window around each anchor
                  frame, T >= 2*max_shift + 1
      center_idx: (B,) index of the anchor frame within each window (usually
                  a constant, e.g. T // 2, unless your windows are ragged
                  near sequence boundaries)

    No SkeletonAugment is applied — the two SwAV views are two distinct
    frames sampled from the window via sample_two_temporal_views.
    """
    model.train()
    running_loss = 0.0

    for step, (sequence, center_idx) in enumerate(dataloader):
        sequence = sequence.to(device)
        center_idx = center_idx.to(device)

        x_a, x_b = sample_two_temporal_views(
            sequence, center_idx, max_shift=max_shift, min_sep=min_sep
        )

        z_a, scores_a = model(x_a)
        z_b, scores_b = model(x_b)

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


# --------------------------------------------------------------------------
# 7. Initializing prototypes from your existing GMM / watershed centers
# --------------------------------------------------------------------------

def init_prototypes_from_gmm(model: SwAVSkeletonModel, gmm_means, embed_normalize=True):
    """
    gmm_means: numpy array or tensor (K, D) — means of your existing GMM fit
    on pretrained SkeletonMAE embeddings. K must equal model.prototypes'
    num_prototypes and D must equal embed_dim.
    """
    centers = torch.as_tensor(gmm_means, dtype=torch.float32)
    if embed_normalize:
        centers = F.normalize(centers, dim=1, p=2)
    model.prototypes.init_from_centers(centers)


# --------------------------------------------------------------------------
# Usage sketch
# --------------------------------------------------------------------------
"""
encoder = load_pretrained_skeletonmae_encoder()   # your existing encoder
embed_dim = 128                                    # match your MAE output dim
num_prototypes = 60                                # e.g. match your GMM component count

model = SwAVSkeletonModel(encoder, embed_dim, num_prototypes).to(device)

# optional: warm-start prototypes from your existing GMM clustering
# init_prototypes_from_gmm(model, gmm.means_)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# dataloader must now yield (sequence, center_idx) pairs:
#   sequence:   (B, T, J, C), a window of T=2*max_shift+1 (or wider) frames
#               around each anchor frame
#   center_idx: (B,) index of the anchor frame within the window (often a
#               constant like T // 2 for every sample, unless windows are
#               ragged near sequence boundaries — see caveat below)

max_shift = 2
min_sep = 1  # raise this if consecutive frames look near-identical in your data

for epoch in range(num_epochs):
    freeze = (epoch == 0)  # SwAV convention: freeze prototype grads epoch 0
    avg_loss = train_one_epoch(model, dataloader, optimizer, device=device,
                                max_shift=max_shift, min_sep=min_sep,
                                freeze_prototypes_epoch=freeze)
    print(f"epoch {epoch}: avg loss {avg_loss:.4f}")

# NOTE on window boundaries: sample_two_temporal_views clamps
# center_idx + offset to [0, T-1]. If an anchor frame sits near the start/end
# of its source sequence and your windowing doesn't pad, both offsets can get
# clamped to the same boundary frame, silently defeating min_sep for that
# sample. Easiest fix: build windows only from anchors that have at least
# max_shift frames of context on both sides, or pad sequence boundaries with
# edge-replication before windowing.
"""