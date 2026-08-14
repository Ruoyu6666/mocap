"""
Attach a set of learnable prototype vectors {C_1, C_2， ..., C_K} to your frame-level features. 
Apply two temporal augmentations to a frame sequence (e.g., slight joint jittering or frame dropping), 
compute cluster assignments using the Sinkhorn-Knopp algorithm, and force the representation of frame $t$ under
Augmentation A to predict the prototype assignment of frame $t$ under Augmentation B.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVSkeletonFrameLoss(nn.Module):
    """SwAV-style online clustering loss for frame-level skeleton features.

    Args:
        feature_dim (int): Dimension of input frame representations (d).
        num_prototypes (K): Number of posture/action primitive clusters.
        temperature (float): Softmax temperature for predicted distributions.
        sinkhorn_eps (float): Regularization parameter for Sinkhorn algorithm.
        sinkhorn_iters (int): Number of iterations in Sinkhorn-Knopp.
    """

    def __init__(self,
                 feature_dim: int = 256,
                 num_prototypes: int = 64,
                temperature: float = 0.1,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters

        # Learnable cluster prototypes (K x d)
        self.prototypes = nn.Linear(feature_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def sinkhorn_knopp(self, scores: torch.Tensor) -> torch.Tensor:
        """Solves optimal transport to assign frames to prototype clusters with equal partition constraint."""
        # Exponential similarity matrix Q = exp(scores / eps)
        Q = torch.exp(scores / self.sinkhorn_eps).t()  # Shape: (K, N)
        K, N = Q.shape
        Q /= torch.sum(Q)

        # Sinkhorn normalization iterations
        for _ in range(self.sinkhorn_iters):
            # Normalize rows
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K

            # Normalize columns
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= N

        Q *= N  # Target soft assignment matrix (N x K)
        return Q.t()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Args:

        z1, z2: Normalized frame feature tensors of shape (N, feature_dim) where
        N = Batch_Size * Sequence_Length.
        """
        # 1. L2 normalize frame features & prototype weights
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        with torch.no_grad():
            w = self.prototypes.weight.data
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # 2. Compute prototype projection scores (cosine similarities)
        scores1 = self.prototypes(z1)  # (N, K)
        scores2 = self.prototypes(z2)  # (N, K)

        # 3. Compute optimal target cluster assignments via Sinkhorn-Knopp
        q1 = self.sinkhorn_knopp(scores1)  # (N, K)
        q2 = self.sinkhorn_knopp(scores2)  # (N, K)

        # 4. Predict probability distributions
        p1 = F.log_softmax(scores1 / self.temperature, dim=1)
        p2 = F.log_softmax(scores2 / self.temperature, dim=1)

        # 5. Swapped loss: cross-entropy between (p1, q2) and (p2, q1)
        loss_swapped_1 = -torch.mean(torch.sum(q2 * p1, dim=1))
        loss_swapped_2 = -torch.mean(torch.sum(q1 * p2, dim=1))

        return 0.5 * (loss_swapped_1 + loss_swapped_2)


# --- Usage Example ---
if __name__ == "__main__":
    B, T, D = 8, 30, 256  # 8 sequences, 30 frames each, feature dim 256
    N = B * T

    # Simulated features from 2 augmented views of skeleton sequence
    z1_frames = torch.randn(N, D)
    z2_frames = torch.randn(N, D)

    swav_loss_fn = SwAVSkeletonFrameLoss(
        feature_dim=D, num_prototypes=32
    )
    loss = swav_loss_fn(z1_frames, z2_frames)

    print(f"SwAV Frame Loss: {loss.item():.4f}")