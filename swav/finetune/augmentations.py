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
 
    Adjust joint_pairs for your skeleton's left/right symmetry if you want mirroring; leave mirror_prob=0 to disable.
    """
    def __init__(self,
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

