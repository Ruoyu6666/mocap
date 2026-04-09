
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
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0., type=float, help='stochastic depth decay rate')
    parser.add_argument('--norm_layer', default=nn.LayerNorm, type=type, help='normalization layer')
    parser.add_argument('--norm_skes_loss', action='store_true', help='if True, normalize skeletons before computing loss')
    
    
    """Dataset and DataLoader parameters"""
    parser.add_argument("--dataset",  type=str, default='mocap')
    parser.add_argument("--task",  type=str, default='CLB')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/code/mocap/data/mocap/data_CLB.pkl')
    parser.add_argument("--sliding_window", default=99, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--fill_holes", default=False, type=str2bool)
    parser.add_argument("--split", default=None, type=dict) 
    #parser.add_argument("--cache_path", type=str, default='../data/tmp/mabe_mouse_train.pkl')
    #parser.add_argument("--cache", default=False, type=str2bool) # if true cache processed data or load from cache

    # In foward function of STTFormer
    parser.add_argument('--mask_ratio', default=0.8, type=float, help='Masking ratio (percentage of removed patches).')
    
    """Dataset augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=False, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")       # for mabe mice dataset
    parser.add_argument("--include_testdata", action="store_true")  # for mabe mice dataset

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    
    """Training parameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    """Saving and logging"""
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default=None) # checkpoint path for resuming training
    # model path for computing representation
    parser.add_argument("--model_path", type=str, default="/home/rguo_hpc/myfolder/code/mocap/outputs/checkpoints/CLB/mae_checkpoint_20_061_192.pth")
    
    """Type of job"""
    parser.add_argument("--if_val", type=str2bool, default=False)
    parser.add_argument("--job", type=str, choices=["pretrain", "compute_representations","linprobe", "finetune"])

    return parser.parse_args()




def train(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device,  log_writer=None,  args=None):

    model = model.to(device)
    # load checkpoint if exists
    if args.ckpt_path is not None:
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
        model.train()
        results = {'total_loss': 0}
        
        for batch_idx, (x, _)  in enumerate(tqdm(data_loader, total=len(data_loader))):
        #for batch_idx, (x, _) in enumerate(tqdm(islice(loader_train, 100), total=100)):
            x = x.to(device)
            optimizer.zero_grad()
            loss, pred, mask = model(x, mask_ratio=args.mask_ratio)
            loss.backward()
            optimizer.step()

            results["total_loss"]  += loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = results["total_loss"] / (batch_idx + 1)
                print(loss.item())
                print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx+1}/{len(data_loader)}], Loss: {avg_loss:.4f}")
                #writer.add_scalar('train/loss', avg_loss, epoch * len(data_loader) + batch_idx)
                #results["total_loss"] = 0.0

        avg_total_loss = results["total_loss"] / len(data_loader)
        
        print(f'Epoch {epoch}/{args.epochs} - Loss: {avg_total_loss:.4f},')
        
        # Save checkpoint
        if args.save_dir and ((epoch % 5 == 0 or epoch == args.epochs)):
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'mae_checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    save_model(model, optimizer, args)
    print(f"Model saved at {args.save_dir}/models/")
    save_results(results, args)




if __name__ == "__main__":

    timestamp = readable_timestamp()
    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Set up data set and data loader"""
    if args.dataset == "mocap":
        dataset_train = MocapDataset(mode = args.job,
                                    path_to_data_dir=args.path_to_data_dir,
                                    datasets = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"],
                                    task = args.task , # CLB, FL2 or Tr
                                    sampling_rate=args.sampling_rate,
                                    num_frames=args.num_frames,
                                    sliding_window=args.sliding_window,
                                    fill_holes=args.fill_holes,
                                    augmentations=args.data_augment,
                                    view_invariant = True, 
                                    left_idx = 3,       # default left hip
                                    right_idx = 8,       # default right hip
                                    index_frame = 149,
                                    model = "SkeletonMAE",)
        data_loader = DataLoader(dataset_train, #sampler=sampler_train,
                                 batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=args.pin_mem, drop_last=True,)
    """
    elif args.dataset == "mabe_mice":
        dataset_train = MABeMouseDataset(mode = "pretrain",
                                         path_to_data_dir=args.path_to_data_dir,
                                        sampling_rate=args.sampling_rate,
                                        num_frames=args.num_frames, 
                                        sliding_window=args.sliding_window,
                                        fill_holes=args.fill_holes,
                                        #cache_path=args.cache_path, cache=args.cache,
                                        augmentations=args.data_augment, #centeralign=args.centeralign,
                                        include_testdata=False,)
    """
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

        train(model, data_loader, optimizer, device, log_writer=None, args=args)
    