"""
SwAV-style self-supervised fine-tuning for skeleton frame embeddings.

Assumes you already have:
  - a pretrained SkeletonMAE encoder that maps (B, T, J, C) or (B, J, C) skeleton input to (B, D) frame-level embeddings
  - a dataloader that yields batches of raw skeleton frames (+ optionally frame
    indices / sequence ids, useful for temporal-shift augmentation and for
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


def temporal_shift_view(sequence: torch.Tensor, center_idx: torch.Tensor, max_shift: int = 2):
    """
    Optional: instead of (or in addition to) augmenting the same frame twice,
    pull the second view from a nearby frame in the same sequence. Encourages
    the representation to be smooth over short time windows.

    sequence: (B, T, J, C) — a window of frames per sample
    center_idx: (B,) index of the "anchor" frame within each window
    Returns: (B, J, C) frames at center_idx + random offset in [-max_shift, max_shift]
    """
    B, T, J, C = sequence.shape
    offset = torch.randint(-max_shift, max_shift + 1, (B,), device=sequence.device)
    idx = (center_idx + offset).clamp(0, T - 1)
    return sequence[torch.arange(B), idx]  # (B, J, C)


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

def train_one_epoch(model: SwAVSkeletonModel, dataloader, optimizer,
                    augment: SkeletonAugment, device: str = "cuda",
                    freeze_prototypes_epoch: bool = False, log_every: int = 50,
                   ):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        # batch: (B, J, C) raw frame skeleton coords — adapt to your loader
        x = batch.to(device)

        x_a = augment(x)
        x_b = augment(x)

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
augment = SkeletonAugment(joint_dropout_p=0.1, jitter_std=0.01, rot_deg=15.0)

for epoch in range(num_epochs):
    freeze = (epoch == 0)  # SwAV convention: freeze prototype grads epoch 0
    avg_loss = train_one_epoch(model, dataloader, optimizer, augment,
                                device=device, freeze_prototypes_epoch=freeze)
    print(f"epoch {epoch}: avg loss {avg_loss:.4f}")
"""