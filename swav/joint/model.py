import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap") 
from datasets.augmentations import Augmentations

from swav.finetune.layers import ProjectionHead, PrototypeLayer
from swav.finetune.model import SwAVSkeletonModel




class JointMAESwAVModel(nn.Module):
    """
    Trains a full SkeletonMAE model (encoder + decoder) and a SwAV prototype head TOGETHER: one loss for masked reconstruction, 
    one loss for clustering, both backpropagating into the same shared encoder every step.
 
    Since I don't have your actual SkeletonMAE class, this is built agains a configurable interface rather than assuming exact method names 
    — point feature_method / mae_loss_method at whatever your class actually calls them:
 
      feature_method (default "forward_features"): a method mae_model.<feature_method>(clip) -> (B, T, D) that returns full, 
                                        UNMASKED per-frame features — used for the SwAV side. This is the same convention as 
                                        --encoder_forward_method used elsewhere in this codebase.
 
      mae_loss_method (default "forward_loss"): a method mae_model.<mae_loss_method>(clip, mask_ratio=...) -> loss 
                        or -> (loss, ...) with the reconstruction loss as the FIRST element (common convention in MAE implementations, 
                        e.g. `loss, pred, mask = model(imgs, mask_ratio)`). Anything after the first element is ignored here.
 
    NOTE: mae_model.forward is monkeypatched at construction time to call feature_method, so mae_model(clip) always returns (B, T, D) 
    features — this matches the `.encoder` interface expected by compute_new_representations_clip and other utilities in this file. 
    
    If you need the model's original forward() behavior elsewhere, call it via a saved reference before constructing this wrapper, 
    or call mae_loss_method directly (which is looked up fresh each call, not affected by the forward() patch).

    """
 
    def __init__(self,
                mae_model: nn.Module,
                embed_dim: int,
                num_prototypes: int = 60,
                projection_head: nn.Module = None,
                feature_method: str = "forward_features",
                mae_loss_method: str = "forward_loss",):
        
        super().__init__()
        self.mae_model = mae_model
        self.mae_loss_method = mae_loss_method
        self.projection_head = projection_head
        proto_in_dim = (projection_head.net[-1].out_features if projection_head is not None else embed_dim)
        self.prototypes = PrototypeLayer(proto_in_dim, num_prototypes)
 
        if feature_method != "forward":
            method = getattr(mae_model, feature_method, None)
            if method is None:
                raise AttributeError(f"mae_model has no method '{feature_method}'")
            mae_model.forward = method
 
    @property
    def encoder(self):
        # alias so compute_new_representations_clip and similar utilities
        # that call model.encoder(clip) work unchanged
        return self.mae_model
 
    def reconstruction_loss(self, clip: torch.Tensor, mask_ratio: float):
        method = getattr(self.mae_model, self.mae_loss_method)
        out = method(clip, mask_ratio=mask_ratio)
        return out[0] if isinstance(out, (tuple, list)) else out
 



def train_one_epoch_joint_mae_swav(
    model: JointMAESwAVModel,
    dataloader,
    optimizer,
    augment: Augmentations,
    device: str = "cuda",
    mask_ratio: float = 0.75,
    recon_weight: float = 1.0,
    swav_weight: float = 1.0,
    freeze_prototypes_epoch: bool = False,
    log_every: int = 50,
):
    """
    Each step: (1) masked-reconstruction loss on the raw clip via mae_loss_method, 
    (2) SwAV loss on two independently-augmented full views via feature_method (same augmented-view scheme as
    train_one_epoch_clip_augmented). Both losses backprop into the shared encoder in the same optimizer step 
    — recon_weight / swav_weight let you balance their relative influence.
 
    This costs THREE forward passes per batch through the encoder path (one for the masked-reconstruction pass, 
    two for the augmented SwAV views) — more expensive than the SwAV-only training loops, unavoidable
    since reconstruction needs the decoder and SwAV needs full features.
    """
    model.train()
    running_total, running_recon, running_swav = 0.0, 0.0, 0.0
 
    for step, batch in enumerate(dataloader):
        clip = batch[0] if isinstance(batch, (tuple, list)) else batch
        clip = clip.to(device)  # (B, T, J, C)
 
        # 1. masked reconstruction loss (uses the decoder, via mae_loss_method)
        recon_loss = model.reconstruction_loss(clip, mask_ratio=mask_ratio)
 
        # 2. SwAV loss on two augmented full-clip views (feature_method only)
        clip_a = augment(clip)
        clip_b = augment(clip)
        z_seq_a = model.encoder(clip_a)  # (B, T, D) — encoder.forward is feature_method
        z_seq_b = model.encoder(clip_b)
 
        B, T, D = z_seq_a.shape
        z_a = z_seq_a.reshape(B * T, D)
        z_b = z_seq_b.reshape(B * T, D)
 
        p_a = model.projection_head(z_a) if model.projection_head is not None else z_a
        p_b = model.projection_head(z_b) if model.projection_head is not None else z_b
        p_a = F.normalize(p_a, dim=1, p=2)
        p_b = F.normalize(p_b, dim=1, p=2)
 
        scores_a = model.prototypes(p_a)
        scores_b = model.prototypes(p_b)
        swav_l = swav_loss(scores_a, scores_b)
 
        total_loss = recon_weight * recon_loss + swav_weight * swav_l
 
        optimizer.zero_grad()
        total_loss.backward()
 
        if freeze_prototypes_epoch:
            for p in model.prototypes.parameters():
                p.grad = None
 
        optimizer.step()
        model.prototypes.normalize_prototypes()
 
        running_total += total_loss.item()
        running_recon += recon_loss.item()
        running_swav += swav_l.item()
        if step % log_every == 0:
            print(
                f"step {step:5d}  total {total_loss.item():.4f}  "
                f"recon {recon_loss.item():.4f}  swav {swav_l.item():.4f}"
            )
 
    n = max(1, len(dataloader))
    return running_total / n, running_recon / n, running_swav / n