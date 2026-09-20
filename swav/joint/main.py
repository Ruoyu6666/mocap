import os
import json
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap") 
from datasets.sdannce import SdannceDataset

from swav.joint.model import JointMAESwAVModel
from swav.joint.engine import train_one_epoch_joint_pretrain 
from swav.joint.utils import compute_representations
from swav.finetune.layers import ProjectionHead, ClassifierHead
from swav.finetune.utils import build_optimizer, init_prototypes_from_gmm


def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


fmr1_fold_1 = {"train":[402, 404, 405, 406, 407, 408], "valid": [401, 403]}
fmr1_fold_2 = {"train":[401, 403, 405, 406, 407, 408], "valid": [402, 404]}
fmr1_fold_3 = {"train":[401, 402, 403, 404, 407, 408], "valid": [405, 406]}
fmr1_fold_4 = {"train":[401, 402, 404, 405, 406, 407], "valid": [403, 408]}



def build_mae(args) -> nn.Module:
    module = importlib.import_module(args.mae_module)
    mae_cls = getattr(module, args.mae_class)
    kwargs = dict(
        dim_in=args.dim_in,
        dim_feat=args.dim_feat,
        decoder_dim_feat=args.decoder_dim_feat,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_frames=args.num_frames,
        num_joints=args.num_joints,
        patch_size=args.patch_size,
        t_patch_size=args.t_patch_size,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        rope_ratio=args.rope_ratio,
        drop_path_rate=args.drop_path_rate,
        norm_skes_loss=args.norm_skes_loss,
        dataset=args.dataset,
        protocol = args.protocol
    )
    if args.mae_kwargs:
        kwargs.update(json.loads(args.mae_kwargs))
    mae = mae_cls(**kwargs)
 
    if not hasattr(mae, "forward_encoder_full"):
        raise AttributeError(f"{args.mae_class} has no forward_encoder_full method — add it "
            f"per pretrain_joint_mae_swav.py's module docstring before using this script.")
 
    if args.mae_ckpt:
        state_dict = torch.load(args.mae_ckpt, map_location="cpu", weights_only=False)["model"]
        mae.load_state_dict(state_dict)
        #missing, unexpected = mae.load_state_dict(state_dict, strict=False)
        #if missing or unexpected:
        #    print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")
 
    return mae


# --------------------------------------------------------------------------
# Argparser
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Joint SkeletonMAE + SwAV pretraining")
    p.add_argument("--path_to_data_dir", type=Path, default="/home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl")
    p.add_argument("--sliding_window", type=int, default=24)
    # model loading
    p.add_argument("--mae_module", type=str, default= "swav.joint.model", help="python import path holding your SkeletonMAE class")
    p.add_argument("--mae_class", type=str, default="SkeletonMAE")
    p.add_argument("--mae_ckpt", type=str, default=None, help="optional state_dict checkpoint to initialize the SkeletonMAE weights from")
    p.add_argument("--mae_kwargs", type=str, default=None, help="optional JSON string to override/extend the SkeletonMAE kwargs below")
 
    # SkeletonMAE constructor arguments
    p.add_argument("--dim_in", type=int, default=3)
    p.add_argument("--dim_feat", type=int, default=192)
    p.add_argument("--decoder_dim_feat", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--decoder_depth", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--num_joints", type=int, default=18, help="must match clips.npy's V dimension")
    p.add_argument("--patch_size", type=int, default=3)
    p.add_argument("--t_patch_size", type=int, default=1)
    p.add_argument("--qkv_bias", type=str2bool, default=True)
    p.add_argument("--qk_scale", type=float, default=None)
    p.add_argument("--drop_rate", type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.0)
    p.add_argument("--rope_ratio", type=float, default=1)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--norm_skes_loss", type=str2bool, default=False)
    p.add_argument("--dataset", type=str, default="sdannce")
 
    # SwAV head
    p.add_argument("--num_prototypes", type=int, default=128)
    p.add_argument("--proj_hidden_dim", type=int, default=256)
    p.add_argument("--proj_out_dim", type=int, default=None)
    p.add_argument("--gmm_means_path", type=str, default=None, help="optional .npy path (K, proj_out_dim) to warm-start prototypes")
 
    # joint training
    p.add_argument("--segment_mask_ratio", type=float, default=0.5)
    p.add_argument("--seg_len", type=int, default=5)
    p.add_argument("--joint_mask_ratio", type=float, default=0.5)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--swav_weight", type=float, default=0.1, help="reconstruction loss tends to dominate early " \
                    "— start this low relative to recon_weight and increase if the SwAV loss curve is flat")
    p.add_argument("--pool", type=str, default="mean", help="how to pool token sequences to clip-level vectors for the SwAV loss")
 
    # optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle",type=str2bool, default=False)
 
    # misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./pretrain_output")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5)
 
    # representation extraction
    p.add_argument("--protocol", default="pretrain")
    p.add_argument("--representation_which", type=str, default="raw",  choices=["raw", "projection", "cluster"],
                    help="'raw': encoder features, hand these off to the SwAV fine-tuning stage. "
                         "'projection'/'cluster': THIS stage's own SwAV head outputs.")
    # checkpoint loading
    p.add_argument("--checkpoint_path", type=str, default=None, help="load a full JointMAESwAVModel checkpoint"
                    " (mae + projection_head + prototypes together) before proceeding")
    return p


def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- skeletonMAE + Model -----
    mae = build_mae(args)
    model = JointMAESwAVModel(mae, num_prototypes=args.num_prototypes,
                proj_hidden_dim=args.proj_hidden_dim, proj_out_dim=args.proj_out_dim,
                ).to(args.device)
    ckpt =  None
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)

    if args.protocol =="compute_representations":
        dataset_train= SdannceDataset(path_to_data_dir=args.path_to_data_dir, 
                                      num_frames=args.num_frames, sliding_window=5,
                                      split = fmr1_fold_1, if_val = False)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                            shuffle=False, drop_last=False, num_workers=args.num_workers)

        dataset_valid= SdannceDataset(path_to_data_dir=args.path_to_data_dir, 
                                      num_frames=args.num_frames, sliding_window=5, 
                                      split = fmr1_fold_1, if_val = True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, 
                            shuffle=False, drop_last=False, num_workers=args.num_workers,)
        print(f"computing representations (which='{args.representation_which}'")

        new_repr_tr = compute_representations(model, dataloader_train, 
                                args.device, args.representation_which, args.t_patch_size)
        repr_path_tr = os.path.join(args.output_dir, "new_representations_train.npy")
        np.save(repr_path_tr, new_repr_tr.numpy())
            
        new_repr_val = compute_representations(model, dataloader_valid, 
                                        args.device, args.representation_which, args.t_patch_size)
        repr_path_val = os.path.join(args.output_dir, "new_representations_valid.npy")
        np.save(repr_path_val, new_repr_val.numpy())

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if ckpt is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[info] restored optimizer state from checkpoint")

        if args.gmm_means_path:
            gmm_means = np.load(args.gmm_means_path)
            init_prototypes_from_gmm(model, gmm_means)
            print(f"Warm-started prototypes from {args.gmm_means_path}")

        dataset = SdannceDataset(path_to_data_dir=args.path_to_data_dir,
                        num_frames=args.num_frames, sliding_window=args.sliding_window,
                        augmentations=False, split = None, if_val = False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        start_epoch = ckpt["epoch"] if ckpt is not None else 0
        log_path = os.path.join(args.output_dir, "loss_log.csv")
        with open(log_path, "w") as f:
            f.write("epoch,avg_total,avg_recon,avg_swav\n")

        for epoch in range(start_epoch, args.epochs):
            freeze_prototypes = (epoch == start_epoch)
            total, recon, sw = train_one_epoch_joint_pretrain(
                model, dataloader, optimizer, device=args.device,
                pool=args.pool, segment_mask_ratio=args.segment_mask_ratio, 
                seg_len=args.seg_len, joint_mask_ratio=args.joint_mask_ratio,
                recon_weight=args.recon_weight, swav_weight=args.swav_weight,
                freeze_prototypes_epoch=freeze_prototypes, log_every=args.log_every,
            )
            print(f"epoch {1+epoch:4d}  total {total:.4f}  recon {recon:.4f}  swav {sw:.4f}")
            with open(log_path, "a") as f:
                f.write(f"{1+epoch},{total:.6f},{recon:.6f},{sw:.6f}\n")
 
            if (epoch + 1) % args.save_every == 0 or epoch == start_epoch + args.epochs - 1:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({"epoch": epoch + 1, 
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),}, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")    

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)