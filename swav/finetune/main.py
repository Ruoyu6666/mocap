import argparse
import importlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from retrain.swav.layers import ProjectionHead, PrototypeLayer
from retrain.swav.model import SwAVSkeletonModel
from retrain.swav.datasets import build_dataset, ClipDataset
from retrain.swav.utils import build_optimizer, compute_new_representations_clip, init_prototypes_from_gmm
from retrain.swav.engine import train_one_epoch_clip_binned

# --------------------------------------------------------------------------
# Encoder loading
# --------------------------------------------------------------------------

def build_encoder(args) -> nn.Module:
    module = importlib.import_module(args.encoder_module)
    encoder_cls = getattr(module, args.encoder_class)

    kwargs = dict(
        dim_in=args.dim_in,
        num_classes=args.num_classes,
        dim_feat=args.dim_feat,
        depth=args.depth,
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
        drop_path_rate=args.drop_path_rate,
        protocol=args.protocol,
        dataset=args.dataset,
    )
    encoder = encoder_cls(**kwargs)

    if args.encoder_ckpt:
        # allow either a raw state_dict or a checkpoint dict with a 'state_dict' / 'model' key
        state_dict = torch.load(args.encoder_ckpt, map_location=args.device, weights_only=False)["model"]
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")

    return encoder


# --------------------------------------------------------------------------
# Argparser
# --------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="SwAV clip-based fine-tuning for SkeletonMAE encoders")

    # data
    p.add_argument("--clips_path", type=str, default = "./others/clips.npy", help="path to .npy file of clips, shape (N, T, J, C)")
    p.add_argument("--center_idx_path", type=str, default=None, help="optional .npy file of per-clip anchor indices, shape (N,). Defaults to T//2.")

    # encoder
    p.add_argument("--encoder_module", type=str, default= "models.skeletonMAE.model.encoder", help="python import path of encoder clas")
    p.add_argument("--encoder_class", type=str, default= "STTFEncoder", help="class name within encoder_module, e.g. 'SkeletonMAEEncoder'")
    p.add_argument("--encoder_ckpt", type=str, default="/home/rguo_hpc/myfolder/mocap/outputs/50patch3/checkpoints/mae_checkpoint_epoch_20.pth", 
                   help="optional path to a state_dict checkpoint for pretrained encoder weights")
    # STTFEncoder constructor arguments
    p.add_argument("--dim_in", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--dim_feat", type=int, default=192, help="encoder embedding dim D — also used as --embed_dim")
    p.add_argument("--depth",  type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--num_frames", type=int, default=50, help="clip length in raw frames — must match clips.npy's T dimension")
    p.add_argument("--num_joints", type=int, default=18, help="must match clips.npy's J dimension")
    p.add_argument("--patch_size", type=int, default=3)
    p.add_argument("--t_patch_size", type=int, default=1)
    p.add_argument('--qkv_bias', action='store_true', help='if True, add a learnable bias to query, key, value')
    p.add_argument("--qk_scale", type=float, default=None)
    p.add_argument("--drop_rate", type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.01)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--protocol", type=str, default="compute_representations")
    p.add_argument("--dataset", type=str, default="sdannce")


    # SwAV model
    p.add_argument("--num_prototypes", type=int, default=60)
    p.add_argument("--mode", type=str, default="finetune", choices=["finetune", "freeze", "finetune_last_n"])
    p.add_argument("--unfreeze_n", type=int, default=2, help="only used when --mode finetune_last_n")
    p.add_argument("--encoder_blocks_attr", type=str, default="blocks", help="attribute name on your encoder holding its transformer block list")
    p.add_argument("--use_projection_head", action="store_true", help="add a trainable projection head between encoder output and prototypes ")
    p.add_argument("--proj_hidden_dim", type=int, default=256)
    p.add_argument("--proj_out_dim", type=int, default=None)
    p.add_argument("--gmm_means_path", type=str, default=None, help="optional .npy path (K, D) to warm-start prototypes from existing GMM cluster centers")

    # SwAV training
    p.add_argument("--pairing_mode", type=str, default="binned", choices=["anchor_window", "binned"],
                    help="'anchor_window': sample n_views frames within max_shift of each clip's center_idx (needs center_idx from the dataset).")
    p.add_argument("--n_views", type=int, default=2, help="[anchor_window only] number of temporal views sampled per clip (>=2). ")
    p.add_argument("--max_shift", type=int, default=2, help="[anchor_window only] max frame offset from the anchor for any sampled view")
    p.add_argument("--n_bins", type=int, default=5, help="[binned only] number of contiguous segments to split each clip into")
    p.add_argument("--min_sep", type=int, default=4, help="minimum frame separation enforced between sampled views")

    # optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./swav_output")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2,help="save a checkpoint every N epochs, in addition to the final one")
    p.add_argument("--compute_representations", action="store_true", help="after training, run compute_new_representations_clip " \
                                                "over the full dataset (unshuffled) and save the result to output_dir")
    p.add_argument("--representation_which", type=str, default="projection",choices=["projection", "raw", "cluster"])
    p.add_argument("--representation_frame_selection", type=str, default="all", choices=["all", "center"], 
                   help="'all': one row per frame in every clip. 'center': one row per clip, taken at its center_idx — only valid with anchor_window pairing.")

    p.add_argument("--checkpoint_path", type=str, default=None,)


    return p



def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = build_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,)
    full_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,)

    encoder = build_encoder(args)

    projection_head = None
    if args.use_projection_head:
        projection_head = ProjectionHead(in_dim=args.dim_feat, hidden_dim=args.proj_hidden_dim, out_dim=args.proj_out_dim)

    model = SwAVSkeletonModel(encoder, args.dim_feat, args.num_prototypes, mode=args.mode, 
                              unfreeze_n=args.unfreeze_n, encoder_blocks_attr=args.encoder_blocks_attr, 
                              projection_head=projection_head,).to(args.device)

    if args.gmm_means_path:
        gmm_means = np.load(args.gmm_means_path)
        init_prototypes_from_gmm(model, gmm_means)
        print(f"Warm-started prototypes from {args.gmm_means_path}")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    log_path = os.path.join(args.output_dir, "loss_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,avg_loss\n")

    for epoch in range(args.epochs):
        freeze_prototypes = (epoch == 0)  # standard SwAV convention
        if args.pairing_mode == "binned":
            avg_loss = train_one_epoch_clip_binned(model, dataloader, optimizer, 
                    device=args.device, n_bins=args.n_bins, min_sep=args.min_sep,
                    freeze_prototypes_epoch=freeze_prototypes, log_every=args.log_every,)
        """
        else:
            avg_loss = train_one_epoch_clip(model, dataloader, optimizer, device=args.device,
                n_views=args.n_views, max_shift=args.max_shift, min_sep=args.min_sep,
                freeze_prototypes_epoch=freeze_prototypes, log_every=args.log_every,)
        """
        print(f"epoch {epoch:4d}  avg_loss {avg_loss:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),}, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")




    if args.compute_representations:
        frame_selection = args.representation_frame_selection
        if args.pairing_mode == "binned" and frame_selection == "center":
            print("[warn] --pairing_mode binned has no center_idx; forcing representation_frame_selection='all'")
            frame_selection = "all"

        print(f"computing final representations (which='{args.representation_which}', frame_selection='{frame_selection}')...")
        new_repr, clip_ids, frame_ids = compute_new_representations_clip(
            model, full_dataloader, device=args.device,
            which=args.representation_which, frame_selection=frame_selection,
            return_indices=True,
        )
        repr_path = os.path.join(args.output_dir, "new_representations.npy")
        np.save(repr_path, new_repr.numpy())
        print(f"saved {repr_path} with shape {tuple(new_repr.shape)}")
        """
        clip_idx_path = os.path.join(args.output_dir, "clip_index.npy")
        frame_idx_path = os.path.join(args.output_dir, "frame_index.npy")
        np.save(clip_idx_path, clip_ids.numpy())
        np.save(frame_idx_path, frame_ids.numpy())
        print(f"saved {clip_idx_path} / {frame_idx_path} (row i of new_representations.npy came from clip clip_ids[i], frame frame_ids[i])")
        """

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)