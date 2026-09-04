import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from swav.finetune.layers import ProjectionHead, PrototypeLayer


def p_requires_grad_any(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())

# --------------------------------------------------------------------------
# 5. Wrapper module: encoder + prototypes
# --------------------------------------------------------------------------
class SwAVSkeletonModel(nn.Module):
    """
    mode:
      "finetune"        - encoder fully trainable
      "freeze"          - encoder frozen, only the prototype layer (and the clustering of existing representation) is learned
      "finetune_last_n" - freeze all encoder params except the last `unfreeze_n` transformer blocks, given via `encoder_blocks_attr` 
                         (a middle ground: let high-level features adapt while keep most of the pretrained representation fixed)
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 embed_dim: int, 
                 num_prototypes: int = 60,
                 mode: str = "finetune", 
                 unfreeze_n: int = 2, 
                 encoder_blocks_attr: str = "blocks", 
                 projection_head: nn.Module = None,):
        super().__init__()
        assert mode in ("finetune", "freeze", "finetune_last_n")
        self.mode = mode
        self.encoder = encoder

        # projection_head is always trainable. Pass a ProjectionHead to use it; prototypes then operate on its output dim, NOT embed_dim
        self.projection_head = projection_head
        proto_in_dim = projection_head.net[-1].out_features if projection_head is not None else embed_dim
        self.prototypes = PrototypeLayer(proto_in_dim, num_prototypes)
 
        if mode == "freeze":
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
 
        elif mode == "finetune_last_n":
            for p in self.encoder.parameters():
                p.requires_grad = False
            blocks = getattr(self.encoder, encoder_blocks_attr, None)
            assert blocks is not None, (f"encoder has no attribute '{encoder_blocks_attr}' — pass the correct"
                                        f"attribute name holding your transformer block list via encoder_blocks_attr")
            # unfrozen blocks
            for block in blocks[-unfreeze_n:]:
                for p in block.parameters():
                    p.requires_grad = True 

            
    def train(self, mode: bool = True):
        # override train() so encoder stays in eval() when frozen, even if model.train() called on the whole wrapper (e.g. at each epoch start)
        super().train(mode)
        if self.mode == "freeze":
            self.encoder.eval()
        elif self.mode == "finetune_last_n":
            self.encoder.eval()
            blocks = getattr(self.encoder, "blocks", None)
            if blocks is not None:
                for i, block in enumerate(blocks):
                    if p_requires_grad_any(block):
                        block.train(mode)
        return self
"""
    def forward(self, x: torch.Tensor):
        # Single-frame-input path:  x is (B, J, C) -> (B, D), one embedding per sample. (i.e. NOT your clip-based encoder). 
        # Kept for the single-frame variant (train_one_epoch / sample_two_temporal_views). 
        if self.mode == "freeze":
            with torch.no_grad():
                z = self.encoder(x)              # (B, D), frozen, never changes
        else:
            z = self.encoder(x)
        p = self.projection_head(z) if self.projection_head is not None else z
        p = F.normalize(p, dim=1, p=2)
        scores = self.prototypes(p)      # (B, K)
        return z, p, scores
    
    def forward_clip(self, clip: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor):
"""
"""
        Clip-input path: matches an encoder that maps (B, T, J, C) -> (B, T, D)
        clip:          (B, T, J, C)
        idx_a, idx_b:  (B,) time indices (per-sample) selecting which of the T frame embeddings to use as the two SwAV views.
        Returns: z_a, z_b (B, D) raw frame embeddings, p_a, p_b (B, D_head) projected+normalized embeddings, 
                scores_a, scores_b (B, K) prototype logits for each view.
"""
"""     if self.mode == "freeze":
            with torch.no_grad():
                z_seq = self.encoder(clip)   # (B, T, D)
        else:
            z_seq = self.encoder(clip)       # (B, T, D)
 
        arange_b = torch.arange(z_seq.shape[0], device=z_seq.device)
        z_a = z_seq[arange_b, idx_a]  # (B, D)
        z_b = z_seq[arange_b, idx_b]  # (B, D)
 
        p_a = self.projection_head(z_a) if self.projection_head is not None else z_a
        p_b = self.projection_head(z_b) if self.projection_head is not None else z_b
        p_a = F.normalize(p_a, dim=1, p=2)
        p_b = F.normalize(p_b, dim=1, p=2)
 
        scores_a = self.prototypes(p_a)
        scores_b = self.prototypes(p_b) 
        return z_a, z_b, p_a, p_b, scores_a, scores_b

 
    def embed_clip(self, clip: torch.Tensor):
"""
"""     Run the encoder (+ projection head) over a whole clip and return per-frame outputs for ALL T frames at once — useful 
        when you want embeddings for every frame, not just two sampled views (e.g. at inference/representation-extraction time). 
        Returns: z_seq (B, T, D) raw, p_seq (B, T, D_head) projected+normalized
"""
"""     if self.mode == "freeze":
            with torch.no_grad():
                z_seq = self.encoder(clip)   # (B, T, D)
        else:
            z_seq = self.encoder(clip)
 
        if self.projection_head is not None:
            B, T, D = z_seq.shape
            p_seq = self.projection_head(z_seq.reshape(B * T, D)).reshape(B, T, -1)
        else:
            p_seq = z_seq
        p_seq = F.normalize(p_seq, dim=-1, p=2)

        return z_seq, p_seq
"""