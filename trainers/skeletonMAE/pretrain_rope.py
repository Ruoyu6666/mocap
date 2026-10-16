import os
import sys
sys.path.append(os.getcwd()) # Adds the current directory to the Python path
import math
import argparse
from tqdm import tqdm
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from trainers.utils import *
from models.skeletonMAE.model.skeletonMAE_rope import SkeletonMAE
from datasets.sdannce import SdannceDataset
from datasets.mabe_mice import MABeMouseDataset
from datasets.mocap import MocapDataset
from utils import adjust_learning_rate


def get_args_parser():
    parser = argparse.ArgumentParser("STTF Training & Compute Representation", add_help=False)
    """SkeletonMAE Model Hyperparameters"""
    parser.add_argument('--dim_in', default=3, type=int, help='input dimension')
    parser.add_argument('--dim_feat', default=192, type=int, help='feature dimension')
    parser.add_argument('--decoder_dim_feat', default=192, type=int, help='decoder feature dimension')
    parser.add_argument('--depth', default=6, type=int, help='number of layers in the encoder')
    parser.add_argument('--decoder_depth', default=1, type=int, help='number of layers in the decoder')
    parser.add_argument('--num_heads', default=8,  type=int, help='number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--num_frames', default=300, type=int, help='number of frames in the input skeleton sequence')
    parser.add_argument('--num_joints', default=10, type=int, help='number of joints in the input skeleton sequence')
    parser.add_argument('--patch_size', default=1, type=int, help='spatial patch size (number of joints per patch)')
    parser.add_argument('--t_patch_size', default=3, type=int, help='temporal patch size (number of frames per patch)')
    
    parser.add_argument('--qkv_bias', action='store_true', help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--qk_scale', default=None, type=float, help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--drop_rate', default=0., type=float, help='dropout rate')
    parser.add_argument('--attn_drop_rate', default=0.01, type=float, help='attention dropout rate')
    parser.add_argument('--rope_ratio', default=0.5, type=float)
    parser.add_argument('--drop_path_rate', default=0., type=float, help='stochastic depth decay rate')
    parser.add_argument('--norm_layer', default=nn.LayerNorm, type=type, help='normalization layer')
    parser.add_argument('--norm_skes_loss', action='store_true', help='if True, normalize skeletons before computing loss')
    
    """Dataset and DataLoader parameters"""
    parser.add_argument("--dataset",  type=str, default='mocap')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/data/mocap/data_FL2.pkl')
    parser.add_argument("--sliding_window", default=30, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--interp_holes", default=False, type=str2bool)
    #parser.add_argument("--split", default=None, type=dict) 
    # parser.add_argument("--if_val", type=str2bool, default=False)

    # In foward function of STTFormer
    parser.add_argument('--segment_mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--joint_mask_ratio', default=0.6, type=float, help='Masking ratio (percentage of removed patches).')

    """Dataset augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=False, type=str2bool)
    parser.add_argument("--view_invariant", default=True, type=str2bool)
    parser.add_argument("--normalize", default=True, type=str2bool)
    
    parser.add_argument("--if_rotate_xz", default=False, type=str2bool)
    parser.add_argument("--centeralign", default=False, type=str2bool)       # for mabe mice dataset
    parser.add_argument("--include_testdata", default=False, type=str2bool)  # for mabe mice dataset

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    
    """Training parameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    """Saving and logging"""
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default=None) # checkpoint path for resuming training
    parser.add_argument("--if_test", type=str2bool, default=False, help="Whether to run test after each training epoch.")

    parser.add_argument("--job", type=str, default="train")
    return parser.parse_args()




def train_one_epoch(model: torch.nn.Module, loader_train: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler=None, log_writer=None, args=None):
    model.train()
    total_loss = 0.0
    header = f"Epoch [{epoch}/{args.epochs}]"
    pbar = tqdm(loader_train, desc=header, total=len(loader_train))
    
    for batch_idx, (x, _)  in enumerate(pbar):
        adjust_learning_rate(optimizer, (epoch - 1) + batch_idx / len(loader_train), args)
        x = x.to(device)
        optimizer.zero_grad()
        loss, pred, mask = model(x, segment_mask_ratio=args.segment_mask_ratio, joint_mask_ratio=args.joint_mask_ratio)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx+1}/{len(loader_train)}], "
                  f"Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            #writer.add_scalar('train/loss', avg_loss, epoch * len(data_loader) + batch_idx)

    avg_total_loss = total_loss / len(loader_train)
    print(f'Epoch {epoch}/{args.epochs} - Train Loss: {avg_total_loss:.4f},')
    return avg_total_loss


@torch.no_grad()
def test(model: torch.nn.Module, loader_test: Iterable, device: torch.device, log_writer=None, args=None):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(loader_test, total=len(loader_test))):
            x = x.to(device)
            loss, _, _ = model(x, mask_ratio=args.mask_ratio)
            total_loss += loss.item()
            n_samples += x.size(0)

    avg_total_loss = total_loss / max(n_samples, 1)
    print(f'Test Loss: {avg_total_loss:.4f}')
    return avg_total_loss


def main(args):
    """Set up dataset & dataloader"""
    if args.dataset == "mocap": 
        dataset_train = MocapDataset(mode = args.job,
                                    path_to_data_dir=args.path_to_data_dir,
                                    datasets = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"],
                                    sampling_rate=args.sampling_rate,
                                    num_frames=args.num_frames,
                                    sliding_window=args.sliding_window,
                                    interp_holes=args.interp_holes,
                                    augmentations=args.data_augment,
                                    view_invariant = args.view_invariant, 
                                    left_idx = 3, right_idx = 8,       # default left, right hip
                                    index_frame = 149,
                                    model = "SkeletonMAE",
                                    split=None,
                                    if_val=False)
        if args.if_test:
            dataset_test = MocapDataset(mode = args.job,
                                        path_to_data_dir=args.path_to_data_dir,
                                        datasets = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"],
                                        sampling_rate=args.sampling_rate,
                                        num_frames=args.num_frames,
                                        sliding_window=args.sliding_window,
                                        interp_holes=args.interp_holes,
                                        augmentations=args.data_augment,
                                        view_invariant = args.view_invariant, 
                                        index_frame = int(args.num_frames/2),
                                        model = "SkeletonMAE",
                                        split=mocap_fold_1,
                                        if_val=True)
    elif args.dataset == "sdannce":
        dataset_train = SdannceDataset(path_to_data_dir=args.path_to_data_dir,
                                       sampling_rate=args.sampling_rate,
                                       num_frames=args.num_frames,
                                       sliding_window=args.sliding_window,
                                       interp_holes=args.interp_holes,
                                       augmentations=args.data_augment,
                                       view_invariant = args.view_invariant,
                                       model = "SkeletonMAE",
                                       split = None, if_val = False)
        if args.if_test:
            dataset_test = SdannceDataset(path_to_data_dir=args.path_to_data_dir,
                                          sampling_rate=args.sampling_rate,
                                          num_frames=args.num_frames,
                                          sliding_window=args.sliding_window,
                                          interp_holes=args.interp_holes,
                                          augmentations=args.data_augment,
                                          view_invariant = args.view_invariant, 
                                          model = "SkeletonMAE",
                                          split = fmr1_fold_1, if_val = True)
    elif args.dataset == "mabe_mice":
        dataset_train = MABeMouseDataset(mode = args.job,
                                         path_to_data_dir=args.path_to_data_dir,
                                         sampling_rate=args.sampling_rate,
                                         num_frames=args.num_frames,
                                         sliding_window=args.sliding_window,
                                         fill_holes=args.interp_holes,
                                         augmentations=args.data_augment,
                                         centeralign = args.centeralign, 
                                         include_testdata = args.include_testdata,
                                         model = "SkeletonMAE")
        if args.if_test:
            dataset_test = MABeMouseDataset(mode = "test",
                                            path_to_data_dir=args.path_to_data_dir,
                                            sampling_rate=args.sampling_rate,
                                            num_frames=args.num_frames,
                                            sliding_window=args.sliding_window,
                                            fill_holes=args.interp_holes,
                                            augmentations=args.data_augment,
                                            centeralign = args.centeralign, 
                                            include_testdata = args.include_testdata,
                                            model = "SkeletonMAE")
    elif args.dataset == "eyetract":
        dataset_train = EyetrackDataset(path_to_data_dir=args.path_to_data_dir, num_frames = args.num_frames)

    
    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, 
                              num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,)
    if args.if_test:
        loader_test = DataLoader(dataset_test, #sampler=sampler_test,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                pin_memory=args.pin_mem, drop_last=False,)

    model = SkeletonMAE(dim_in=args.dim_in, dim_feat=args.dim_feat, decoder_dim_feat=args.decoder_dim_feat,
                        depth=args.depth, decoder_depth=args.decoder_depth,
                        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,  
                        num_frames=args.num_frames, num_joints=args.num_joints,
                        patch_size=args.patch_size, t_patch_size=args.t_patch_size,
                        qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
                        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                        drop_path_rate=args.drop_path_rate, norm_layer=args.norm_layer, 
                        norm_skes_loss=args.norm_skes_loss,dataset=args.dataset,
                        rope_ratio=args.rope_ratio)
    model = model.to(device)
    total_params = sum(p.numel() for p in  model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')
    
    """Set up optimizer and training loop"""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if args.ckpt_path is not None:  # load checkpoint if exists
        print(f"Loading checkpoint from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        print("No checkpoints found, starting training from scratch.")
        os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
        start_epoch = 0
    
    num_epochs = args.epochs - start_epoch
    print('Number of epochs to train:', num_epochs)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_one_epoch(model, loader_train, optimizer, device, epoch, loss_scaler=None, log_writer=None, args=args)
        
        if args.if_test and (epoch % 3 ==0):
            test(model, loader_test, device, log_writer=None, args=args)
        if args.save_dir and ((epoch % 5 == 0 or epoch == args.epochs)):
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'mae_checkpoint_epoch_{epoch}.pth')
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),}, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    save_model(model, optimizer, args)
    print(f"Model saved at {args.save_dir}/models/")


if __name__ == "__main__":
    timestamp = readable_timestamp()
    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)