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




class ClassifierHead(nn.Module):
    """
    Frame-level classification head, attached directly to the encoder's raw per-frame/token output z_seq (B, T, D) — 
    NOT the SwAV projection head's output, since the projection head is specifically shaped for the SwAV cluster space, 
    whereas classification should use the encoder's own (richer, higher-dim) features. Produces per-frame logits 
    (B, T, num_classes); combine with F.cross_entropy(..., ignore_index=...) for partial/sparse labeling.
    """
 
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)
 
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
