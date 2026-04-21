
import os
import sys
sys.path.append(os.getcwd()) # Adds the current directory to the Python path

import argparse
import numpy as np
import pdb
from tqdm import tqdm
from itertools import islice
from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trainers.utils import *
from models.skeletonMAE.util import lr_decay as lrd
from models.skeletonMAE.util import misc as misc
#from models.MAE.util.datasets import build_dataset
from models.skeletonMAE.util.pos_embed import interpolate_temp_embed
#from models.MAE.util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.skeletonMAE.model.skeletonMAE import SkeletonMAE
from models.skeletonMAE.model.encoder import STTFEncoder
from dataset.mabe_mice import MABeMouseDataset
from dataset.mocap import MocapDataset


fold_1 = {
    "CP1A": {"train": ["M14", "M15", "M19"], 
             "valid": ["M1"]},
    "CP1B": {"train": ["M2", "M3", "M4", "M5", "M6"], 
             "valid": ["M1"]},
    "INH1": {"train": ["M2", "M3", "M4", "M5", "M7", "M8", "M9", "M10"],
             "valid": ["M1", "M6"]},
    "INH2": {"train": ["M2", "M3", "M4", "M5", "M7", "M8", "M9", "M10", "M12"],
             "valid": ["M1", "M6", "M11"]},
    "MOS1aD": {"train": ["M5", "M6", "M8", "M9", "M10"],
               "valid": ["M4"]}}

fold_2 = {
    "CP1A": {"train": ["M1", "M15", "M19"], 
            "valid": ["M14"]},
    "CP1B": {"train": ["M1", "M3", "M4", "M5", "M6"], 
            "valid": ["M2"]},
    "INH1": {"train": ["M1", "M3", "M4", "M5", "M6", "M8", "M9", "M10"],
            "valid": ["M2", "M7"]},
    "INH2": {"train": ["M1", "M3", "M4", "M5", "M6", "M8", "M9", "M10", "M11"],
            "valid": ["M2", "M7", "M12"]},
    "MOS1aD": {"train": ["M4", "M6", "M8", "M9", "M10"],
                "valid": ["M5"]}
}
fold_3 = {
    "CP1A": {"train": ["M1", "M14", "M19"], 
            "valid": ["M15"]},
    "CP1B": {"train": ["M1", "M2", "M4", "M5", "M6"], 
            "valid": ["M3"]},
    "INH1": {"train": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M10"],
            "valid": ["M3", "M8"]},
    "INH2": {"train": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M11", "M12"],
            "valid": ["M3", "M8", "M10"]},
    "MOS1aD": {"train": ["M4", "M5", "M8", "M9", "M10"],
                "valid": ["M6"]}
}
fold_4 = {
    "CP1A": {"train": ["M1", "M14", "M15"], 
            "valid": ["M19"]},
    "CP1B": {"train": ["M1", "M2", "M3", "M5", "M6"], 
            "valid": ["M4"]},
    "INH1": {"train": ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M10"],
            "valid": ["M4", "M9"]},
    "INH2": {"train": ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M10", "M12"],
            "valid": ["M4", "M9", "M11"]},
    "MOS1aD": {"train": ["M4", "M5", "M6", "M9", "M10"],
            "valid": ["M8"]}
}




def get_args_parser():

    parser = argparse.ArgumentParser("STTF Training & Compute Representation", add_help=False)

    """SkeletonMAE Model Hyperparameters"""
    parser.add_argument('--dim_in', default=3, type=int, help='input dimension')
    parser.add_argument('--dim_feat', default=192, type=int, help='feature dimension')
    parser.add_argument('--decoder_dim_feat', default=256, type=int, help='decoder feature dimension')
    parser.add_argument('--depth', default=6, type=int, help='number of layers in the encoder')
    parser.add_argument('--decoder_depth', default=1, type=int, help='number of layers in the decoder')
    parser.add_argument('--num_heads', default=8,  type=int, help='number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--num_frames', default=300, type=int, help='number of frames in the input skeleton sequence')
    parser.add_argument('--num_joints', default=10, type=int, help='number of joints in the input skeleton sequence')
    parser.add_argument('--patch_size', default=1, type=int, help='spatial patch size (number of joints per patch)')
    parser.add_argument('--t_patch_size', default=2, type=int, help='temporal patch size (number of frames per patch)')
    parser.add_argument('--qkv_bias', action='store_true', help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--qk_scale', default=None, type=float, help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--drop_rate', default=0., type=float, help='dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0., type=float, help='stochastic depth decay rate')
    parser.add_argument('--norm_layer', default=nn.LayerNorm, type=type, help='normalization layer')
    parser.add_argument('--norm_skes_loss', action='store_true', help='if True, normalize skeletons before computing loss')
    
    
    """Dataset and DataLoader parameters"""
    parser.add_argument("--dataset",  type=str, default='mocap')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/mocap/data/mocap/data_FL2.pkl')
    parser.add_argument("--sliding_window", default=60, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--interp_holes", default=False, type=str2bool)
    #parser.add_argument("--split", default=None, type=dict) 
    # parser.add_argument("--if_val", type=str2bool, default=False)

    # In foward function of STTFormer
    parser.add_argument('--mask_ratio', default=0.7, type=float, help='Masking ratio (percentage of removed patches).')
    
    """Dataset augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=False, type=str2bool)
    parser.add_argument("--view_invariant", default=True, type=str2bool)
    parser.add_argument("--centeralign", default=False, type=str2bool)       # for mabe mice dataset
    parser.add_argument("--include_testdata", action="store_true")  # for mabe mice dataset

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    
    """Training parameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay (default: 0.05)')

    """Saving and logging"""
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default=None) # checkpoint path for resuming training
    
    parser.add_argument("--if_test", type=str2bool, default=False, help="Whether to run test after each training epoch.")
    """Type of job"""
    parser.add_argument("--job", type=str, choices=["pretrain", "compute_representations","linprobe", "finetune"])

    return parser.parse_args()




def train_one_epoch(model: torch.nn.Module, loader_train: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train()
    results = {'total_loss': 0}
    header = f"Epoch [{epoch}/{args.epochs}]"
    pbar = tqdm(loader_train, desc=header, total=len(loader_train))
    
    for batch_idx, (x, _)  in enumerate(pbar):
        x = x.to(device)
        optimizer.zero_grad()
        loss, pred, mask = model(x, mask_ratio=args.mask_ratio)
        loss.backward()
        optimizer.step()

        results["total_loss"]  += loss.item()
        
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = results["total_loss"] / (batch_idx + 1)
            print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx+1}/{len(loader_train)}], Loss: {avg_loss:.4f}")
            #writer.add_scalar('train/loss', avg_loss, epoch * len(data_loader) + batch_idx)

    avg_total_loss = results["total_loss"] / len(loader_train)
    
    print(f'Epoch {epoch}/{args.epochs} - Train Loss: {avg_total_loss:.4f},')




def test(model: torch.nn.Module, loader_test: Iterable, device: torch.device, log_writer=None, args=None):

    model.eval()
    results = {'total_loss': 0}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(loader_test, total=len(loader_test))):
            x = x.to(device)
            loss, _, _ = model(x, mask_ratio=args.mask_ratio)
            results["total_loss"]  += loss.item()

    avg_total_loss = results["total_loss"] / len(loader_test)
    print(f'Test Loss: {avg_total_loss:.4f},')
    







def main(args):
    """Set up data set and data loader"""
    if args.dataset == "mocap":
        dataset_train = MocapDataset(mode = args.job,
                                    path_to_data_dir=args.path_to_data_dir,
                                    datasets = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"],
                                    #task = args.task , # CLB, FL2 or Tr
                                    sampling_rate=args.sampling_rate,
                                    num_frames=args.num_frames,
                                    sliding_window=args.sliding_window,
                                    interp_holes=args.interp_holes,
                                    augmentations=args.data_augment,
                                    view_invariant = args.view_invariant, 
                                    left_idx = 3,       # default left hip
                                    right_idx = 8,       # default right hip
                                    index_frame = 149,
                                    model = "SkeletonMAE",
                                    split=None,
                                    if_val=False)
        loader_train = DataLoader(dataset_train, #sampler=sampler_train,
                                 batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=args.pin_mem, drop_last=True,)
        if args.if_test:
                dataset_test = MocapDataset(mode = args.job,
                                        path_to_data_dir=args.path_to_data_dir,
                                        datasets = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"],
                                        sampling_rate=args.sampling_rate,
                                        num_frames=args.num_frames,
                                        sliding_window=args.sliding_window,
                                        interp_holes=args.interp_holes,
                                        augmentations=False,
                                        view_invariant = True, 
                                        left_idx = 3,       # default left hip
                                        right_idx = 8,       # default right hip
                                        index_frame = 149,
                                        model = "SkeletonMAE",
                                        split=fold_2,
                                        if_val=True)
                loader_test = DataLoader(dataset_test, #sampler=sampler_test,
                                        batch_size=args.batch_size, num_workers=args.num_workers,
                                        pin_memory=args.pin_mem, drop_last=False,)
    if args.job == "pretrain":
        """Set up model for pretrain"""
        model = SkeletonMAE(dim_in=args.dim_in,
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
                            drop_path_rate=args.drop_path_rate, 
                            norm_layer=args.norm_layer, 
                            norm_skes_loss=args.norm_skes_loss,
                            dataset=args.dataset)
        
        total_params = sum(p.numel() for p in  model.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')

    """Set up optimizer and training loop"""
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    model = model.to(device)
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
    for epoch in range(start_epoch + 1, args.epochs+1):
        train_one_epoch(model, loader_train, optimizer, device, epoch, loss_scaler=None, log_writer=None, args=args)
        
        if args.if_test:
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


    