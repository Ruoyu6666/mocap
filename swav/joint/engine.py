from .model import JointMAESwAVModel

import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap") 
from swav.finetune.utils import swav_loss


def train_one_epoch_joint_pretrain(
        model: JointMAESwAVModel, dataloader, optimizer, device, pool: str = "mean",
        segment_mask_ratio: float = 0.5, seg_len: int = 5, joint_mask_ratio: float = 0.5,
        recon_weight: float = 1.0, swav_weight: float = 0.1, 
        freeze_prototypes_epoch: bool = False, log_every: int = 50,):
    """
    Each batch is X1 (or (X1, ...) — extra elements ignored): the same [N, T, V, C]  (or 5-D 
    multi-individual) input your SkeletonMAE.forward() accepts.
    segment_mask_ratio / seg_len / joint_mask_ratio: passed straight through to model.mae.forward_encoder
    """
    model.train()
    running_total, running_recon, running_swav = 0.0, 0.0, 0.0
 
    for step, batch in enumerate(dataloader):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
 
        # mirrors SkeletonMAE.forward()'s own input reshaping
        mae = model.mae
        if mae.dataset == "mabe_mice":
            N, T, M, _ = x.shape
            x = x.reshape(N, T, M, mae.num_joints, mae.dim_in)
        elif x.ndim == 5:
            N, T, M, V, C = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, V, C)
 
        # ----- masked view: reused for BOTH reconstruction and SwAV -----
        latent_masked, mask, ids_restore, TP = mae.forward_encoder(
                            x, segment_mask_ratio, seg_len, joint_mask_ratio)
        pred = mae.forward_decoder(latent_masked, ids_restore, TP)
        recon_loss = mae.forward_loss(x, pred, mask)
        
        # ----- full/clean view: ONE extra encoder pass, SwAV only -----
        latent_full, _ = mae.forward_encoder_full(x)
        scores_masked = model.swav_branch(latent_masked, pool=pool)
        scores_full = model.swav_branch(latent_full, pool=pool)
        loss_swav = swav_loss(scores_full, scores_masked)
 
        loss = recon_weight * recon_loss + swav_weight * loss_swav
        optimizer.zero_grad()
        loss.backward()
 
        if freeze_prototypes_epoch:
            for p in model.prototypes.parameters():
                p.grad = None
        optimizer.step()
        model.prototypes.normalize_prototypes()
 
        running_total += loss.item()
        running_recon += recon_loss.item()
        running_swav += loss_swav.item()
        if step % log_every == 0:
            print(f"step {step:5d}  total {loss.item():.4f}  "
                  f"recon {recon_loss.item():.4f}  swav {loss_swav.item():.4f}")
 
    n = max(1, len(dataloader))
    return running_total / n, running_recon / n, running_swav / n