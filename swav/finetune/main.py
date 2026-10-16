import argparse
import importlib
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap") 
from datasets.augmentations import Augmentations, _resample_time
from datasets.sdannce import SdannceDataset
from datasets.mabe_mice import MABeMouseDataset

from swav.finetune.layers import ProjectionHead, ClassifierHead
from swav.finetune.model import SwAVSkeletonModel
from swav.finetune.utils import str2bool, build_optimizer, compute_new_representations_overlapping, init_prototypes_from_gmm
from swav.finetune.engine import train_one_epoch_clip_augmented, train_one_epoch_clip_augmented_sequence_semisup, train_one_epoch_clip_classifier_only


# --------------------------------------------------------------------------
# Encoder loading
# --------------------------------------------------------------------------
def build_encoder(args) -> nn.Module:
    module = importlib.import_module(args.encoder_module)
    encoder_cls = getattr(module, args.encoder_class)
    kwargs = dict(dim_in=args.dim_in, num_classes=args.num_classes,
                 dim_feat=args.dim_feat, depth=args.depth,
                 num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                 num_frames=args.num_frames, num_joints=args.num_joints,
                 patch_size=args.patch_size, t_patch_size=args.t_patch_size,
                 qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
                 drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, 
                 drop_path_rate=args.drop_path_rate, rope_ratio=1,
                 protocol=args.protocol, dataset=args.dataset)
    encoder = encoder_cls(**kwargs)
    if args.protocol == "finetune":
        if args.encoder_ckpt:
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
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--include_testdata", type=str2bool, default=False)
    # data path
    p.add_argument("--path_to_data_dir", type=Path, default="/home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl")
    p.add_argument("--sampling_rate", type=int, default=1)
    p.add_argument("--sliding_window", type=int, default=60)
    
    # encoder
    p.add_argument("--encoder_module", type=str, default= "models.skeletonMAE.model.encoder_rope", 
                   help="python import path of encoder clas")
    p.add_argument("--encoder_class", type=str, default= "STTFEncoder", help="class name within encoder_module ")
    p.add_argument("--encoder_ckpt", type=str, default="/home/rguo_hpc/myfolder/mocap/outputs/fmr1/50/checkpoints/epoch_10.pth", 
                   help="optional path to a state_dict checkpoint for pretrained encoder weights")
    
    # STTFEncoder constructor arguments
    p.add_argument("--dim_in", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--dim_feat", type=int, default=192, help="encoder embedding dim D — also used as --embed_dim")
    p.add_argument("--depth",  type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--num_frames", type=int, default=50, help="clip length in raw frames — must match clips.npy's T dimension")
    p.add_argument("--num_joints", type=int, default=18, help="must match clips.npy's J dimension")
    p.add_argument("--patch_size", type=int, default=3)
    p.add_argument("--t_patch_size", type=int, default=1)
    p.add_argument('--qkv_bias', type=str2bool, default=False, help='if True, add a learnable bias to query, key, value')
    p.add_argument("--qk_scale", type=float, default=None)
    p.add_argument("--drop_rate", type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.02)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--protocol", choices=["finetune", "compute_representations"], default="finetune")
    p.add_argument("--config", type=str, default = '/home/rguo_hpc/myfolder/mocap/configs/sdannce/fmr1.yaml')

    # SwAV model
    p.add_argument("--num_prototypes", type=int, default=128)
    p.add_argument("--mode", type=str, default="finetune", choices=["finetune", "freeze", "finetune_last_n"])
    p.add_argument("--unfreeze_n", type=int, default=2, help="only used when --mode finetune_last_n")
    p.add_argument("--encoder_blocks_attr", type=str, default="blocks", help="attribute name on your encoder holding its transformer block list")
    p.add_argument("--proj_hidden_dim", type=int, default=256)
    p.add_argument("--proj_out_dim", type=int, default=None)
    p.add_argument("--gmm_means_path", type=str, default=None, help="optional .npy path (K, D) to warm-start prototypes from existing GMM cluster centers")

    p.add_argument("--classifier_hidden_dim", type=int, default=None)
    
    # SwAV training
    p.add_argument("--pairing_mode", type=str, default="augment")
    p.add_argument("--n_views", type=int, default=2, help="[anchor_window only] number of temporal views sampled per clip (>=2). ")

    p.add_argument("--semisup", type=str2bool, default=False)
    p.add_argument("--swav_weight", type=float, default=1.0, help="weight on the SwAV loss term ")
    p.add_argument("--cls_weight", type=float, default=1.0, help="weight on the classification loss term")
    p.add_argument("--ignore_index", type=int, default=-100, help="label value marking an unlabeled frame (matches F.cross_entropy's default)")
    # optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle",type=str2bool, default=True)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./swav_output")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5, help="save a checkpoint every N epochs, in addition to the final one")
    p.add_argument("--checkpoint_path", type=str, default=None, help="saved SwAVSkeletonModel model")
    p.add_argument("--representation_which", type=str, default="raw", choices=["projection", "raw", "cluster"])

    return p




def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Encoder + Model -----
    encoder = build_encoder(args)
    projection_head = None
    if args.proj_out_dim is not None:
        projection_head = ProjectionHead(in_dim=args.dim_feat, hidden_dim=args.proj_hidden_dim, out_dim=args.proj_out_dim)
    classifier_head = None
    
    if args.semisup or args.pairing_mode == "classification_only":
        classifier_head = ClassifierHead(in_dim=args.dim_feat, 
                num_classes=args.num_classes, hidden_dim=args.classifier_hidden_dim,)

    model = SwAVSkeletonModel(encoder, args.dim_feat, args.num_prototypes, mode=args.mode, 
                unfreeze_n=args.unfreeze_n, encoder_blocks_attr=args.encoder_blocks_attr, 
                projection_head=projection_head, classifier_head=classifier_head).to(args.device)
    print("mode, ", model.mode)

    

    # ----- Compute representations only -----
    if args.protocol== "compute_representations":  
        print("load from " + args.checkpoint_path )
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(ckpt, strict=False)
        model = model.to(args.device)
        
        if args.dataset == "mabe":
            dataset_test = MABeMouseDataset(mode="test", path_to_data_dir=args.path_to_data_dir,
                                    sampling_rate=args.sampling_rate, num_frames=args.num_frames,
                                    sliding_window=args.sliding_window, augmentations=False,
                                    include_testdata = args.include_testdata, model = "SkeletonMAE")
            dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers,)
            new_repr_test = compute_new_representations_overlapping(model, dataloader_test, 
                                     args.device, args.representation_which)
            repr_path_val = os.path.join(args.output_dir, "new_representations_test.npy")
            np.save(repr_path_val, new_repr_test.numpy())

        
        if args.dataset == "sdannce":
            with open(args.config) as f:
                split = yaml.safe_load(f)
            dataset_train= SdannceDataset(path_to_data_dir=args.path_to_data_dir, num_frames=args.num_frames, 
                        sliding_window= args.sliding_window, split = split["fold_1"], if_val = False)
            dataset_valid= SdannceDataset(path_to_data_dir=args.path_to_data_dir, num_frames=args.num_frames, 
                        sliding_window=args.sliding_window, split = split["fold_1"], if_val = True)
            dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=False, num_workers=args.num_workers,)
            dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers,)
            
            print(f"computing final representations (which='{args.representation_which}')")
            new_repr_tr = compute_new_representations_overlapping(model, dataloader_train, args.device, args.representation_which)
            repr_path_tr = os.path.join(args.output_dir, "new_representations_train.npy")
            np.save(repr_path_tr, new_repr_tr.numpy())
            
            new_repr_val = compute_new_representations_overlapping(model, dataloader_valid, args.device, args.representation_which)
            repr_path_val = os.path.join(args.output_dir, "new_representations_valid.npy")
            np.save(repr_path_val, new_repr_val.numpy())

    # Train model
    else:
        if args.dataset == "mabe":
            dataset = MABeMouseDataset(mode="train", path_to_data_dir=args.path_to_data_dir,
                        sampling_rate=args.sampling_rate, num_frames=args.num_frames,
                        sliding_window=args.sliding_window, augmentations=False, normalize=False,
                        include_testdata = args.include_testdata, model = "SkeletonMAE")
        if args.dataset == "sdannce":
            dataset= SdannceDataset(path_to_data_dir=args.path_to_data_dir,
                num_frames=args.num_frames, sliding_window=args.sliding_window,
                view_invariant = True, augmentations=False, normalize=False,
                split = None, if_val = False) # augment + normalize later
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        if args.checkpoint_path is not None:
            print("load from " + args.checkpoint_path )
            ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(args.device)
            start_epoch = ckpt["epoch"]
        else:
            start_epoch = 0

        if args.gmm_means_path:
            gmm_means = np.load(args.gmm_means_path)
            init_prototypes_from_gmm(model, gmm_means)
            print(f"Warm-started prototypes from {args.gmm_means_path}")

        optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        log_path = os.path.join(args.output_dir, "loss_log.csv")

        for epoch in range(start_epoch, args.epochs):
            freeze_prototypes = (epoch == 0)  # standard SwAV convention
            if args.pairing_mode == "classification_only":
                avg_loss = train_one_epoch_clip_classifier_only(
                    model, dataloader, optimizer, augment=None, device=args.device,
                    ignore_index=args.ignore_index, log_every=args.log_every,
                )
                print(f"epoch {epoch:4d}  avg_loss {avg_loss:.4f}  ")
                with open(log_path, "a") as f:
                    f.write(f"{epoch},{avg_loss:.6f}\n")

            elif args.pairing_mode == "augment":
                augment = Augmentations()
                if args.semisup:
                    avg_loss, avg_swav, avg_cls = train_one_epoch_clip_augmented_sequence_semisup(
                            model, dataloader, optimizer, augment, device=args.device,
                            swav_weight=args.swav_weight, cls_weight=args.cls_weight,
                            downsample_rates=(1, 5), ignore_index=args.ignore_index,
                            freeze_prototypes_epoch=freeze_prototypes, log_every=args.log_every,)
                    print(f"epoch {epoch:4d}  avg_loss {avg_loss:.4f}  avg_swav {avg_swav:.4f}  avg_cls {avg_cls:.4f}")
                    with open(log_path, "a") as f:
                        f.write(f"{epoch},{avg_loss:.6f},{avg_swav:.6f},{avg_cls:.6f}\n")

                else: 
                    avg_loss = train_one_epoch_clip_augmented(
                            model, dataloader, optimizer, augment, device=args.device,
                            freeze_prototypes_epoch=freeze_prototypes, log_every=args.log_every,
                            dataset=args.dataset, downsample_rates=(1, 5))
                    print(f"epoch {epoch + 1:4d}  avg_loss {avg_loss:.4f}")
                    with open(log_path, "a") as f:
                        f.write(f"{epoch + 1},{avg_loss:.6f}\n")

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({"epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),}, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)