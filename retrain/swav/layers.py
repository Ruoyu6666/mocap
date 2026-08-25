import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Prototype layer
# --------------------------------------------------------------------------
class PrototypeLayer(nn.Module):
    """
    Learnable prototypes C in R^{D x K}, kept L2-normalized on the unit sphere (per SwAV). 
    Call `normalize_prototypes()` after every optimizer.step().
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
        """centers: (K, D) tensor, e.g. GMM means or other centroids computed on existing pretrained embeddings. 
                    Gives the prototypes a behaviorally meaningful head start instead of random init."""
        assert centers.shape == self.prototypes.weight.shape, (f"expected {self.prototypes.weight.shape}, got {centers.shape}")
        centers = F.normalize(centers, dim=1, p=2)
        self.prototypes.weight.copy_(centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.prototypes(z)        # z assumed already L2-normalized




class ProjectionHead(nn.Module):
    """
    Small trainable MLP between the (possibly frozen) encoder output and the prototype layer. Always trainable regardless of encoder mode.
 
    When mode="freeze": since the encoder never gets gradients, head(z) is the only thing that actually changes during training. 
    Its output is a legitimate new frame representation — you choose out_dim — even though the backbone was never touched. 
    Use head(z) (L2-normalized) as your new per-frame embedding instead of, or alongside, the raw frozen z.
    """
 
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim),)
 
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --------------------------------------------------------------------------
# 5. Wrapper module: encoder + prototypes
# --------------------------------------------------------------------------
"""
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
"""