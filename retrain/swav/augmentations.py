
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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





def sample_two_temporal_views(sequence: torch.Tensor, center_idx: torch.Tensor,
                              max_shift: int = 2, min_sep: int = 1,):
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
