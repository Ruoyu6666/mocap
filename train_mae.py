
import os
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

from utils import *
#import models.MAE.util.lr_decay as lrd
#import models.MAE.util.misc as misc
# from .models.MAE.util.datasets import build_dataset
from models.MAE.util.pos_embed import interpolate_temp_embed
#from models.MAE.util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.MAE.model.SkeletonMAE import SkeletonMAE
from models.MAE.model.Encoder import STTFEncoder
from dataset.mabe_mice import MABeMouseDataset



def get_args_parser():

    parser = argparse.ArgumentParser("STTF Training & Compute Representation", add_help=False)

    """SkeletonMAE Model Hyperparameters"""
    parser.add_argument('--dim_in', default=2, type=int, help='input dimension')
    parser.add_argument('--dim_feat', default=128, type=int, help='feature dimension')
    parser.add_argument('--decoder_dim_feat', default=128, type=int, help='decoder feature dimension')
    parser.add_argument('--depth', default=8, type=int, help='number of layers in the encoder')
    parser.add_argument('--decoder_depth', default=3, type=int, help='number of layers in the decoder')
    parser.add_argument('--num_heads', default=8,  type=int, help='number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--num_frames', default=300, type=int, help='number of frames in the input skeleton sequence')
    parser.add_argument('--num_joints', default=12, type=int, help='number of joints in the input skeleton sequence')
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
    parser.add_argument("--dataset",  type=str, default='mabe_mouse')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/data/MaBe/mouse/mouse_triplet_train.npy')
    parser.add_argument("--sliding_window", default=149, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--if_fill_holes", default=False, type=str2bool)
    parser.add_argument("--cache_path", type=str, default='../data/tmp/mabe_mouse_train.pkl')
    parser.add_argument("--cache", default=False, type=str2bool) # if true cache processed data or load from cache

    # In foward function of STTFormer
    parser.add_argument('--mask_ratio', default=0.85, type=float, help='Masking ratio (percentage of removed patches).')
    
    """Dataset augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=True, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument("--include_testdata", action="store_true")

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    
    """Training parameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    #parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    #parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    """Saving and logging"""
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default="./outputs/checkpoints/mae_checkpoint_epoch_2_8layers.pth") # checkpoint path for training

    parser.add_argument("--model_path", type=str, default="./outputs/checkpoints/mae_checkpoint_epoch_2.pth") # model path for computing representation

    """Type of job"""
    parser.add_argument("--job", type=str, choices=["pretrain", "compute_representations"])

    return parser.parse_args()




def train(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
          log_writer=None,  args=None):

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
        results = {'embedding_loss': 0, 
                   'recon_errors': 0, 
                   'total_loss': 0,
                   'perplexities': 0}
        
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
        if args.save_dir and ((epoch % 1 == 0 or epoch == args.epochs)):
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





def compute_representations(model, data_loader, device ,args):
    os.makedirs(args.save_dir + '/representations', exist_ok=True)
    model = model.to(device)
    model.eval()

    all_representations = []

    with torch.no_grad():
        for i, (x, _)  in enumerate(data_loader):
        #for i, (x, _) in enumerate(tqdm(islice(data_loader, 100), total=100)):
            x = x.to(device)
            latent = model(x)      # (N, T, C)
            all_representations.append(torch.squeeze(latent).cpu().numpy())
            if (i + 1) % args.log_interval == 0:
                print(f"Processed {i+1}/{len(data_loader)} batches.")

    all_representations = np.concatenate(all_representations, axis=0)
    
    np.save(args.save_dir + '/representations/mae_representations.npy', all_representations)





if __name__ == "__main__":

    timestamp = readable_timestamp()
    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Set up data set and data loader"""
    dataset_train = MABeMouseDataset(path_to_data_dir=args.path_to_data_dir,
                                     sampling_rate=args.sampling_rate,
                                     num_frames=args.num_frames, 
                                     sliding_window=args.sliding_window,
                                     if_fill=args.if_fill_holes,
                                     #patch_size=args.patch_size,
                                     cache_path=args.cache_path, cache=args.cache,
                                     augmentations=args.data_augment, #centeralign=args.centeralign,
                                     include_testdata=True,)
    

    data_loader = DataLoader(dataset_train, #sampler=sampler_train,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_mem, drop_last=True,)



if args.job == "pretrain":
    """Set up model for pretrain"""
    model = SkeletonMAE(
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
        drop_path_rate=args.drop_path_rate, 
        norm_layer=args.norm_layer, 
        norm_skes_loss=args.norm_skes_loss
    )
    total_params = sum(p.numel() for p in  model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    """Set up optimizer and training loop"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    
    train(model, data_loader, optimizer, device, log_writer=None, args=args)





if args.job == "compute_representations":

    path_test_data = args.path_to_data_dir.replace("_train", "_test")
    
    dataset = MABeMouseDataset(path_to_data_dir=args.path_to_data_dir,
                                sampling_rate=args.sampling_rate,
                                num_frames=args.num_frames, 
                                sliding_window=args.num_frames-1,
                                if_fill=args.if_fill_holes,
                                #patch_size=args.patch_size,
                                cache_path=args.cache_path, cache=False,
                                augmentations=None,
                                include_testdata=True,)

    loader_test = DataLoader(dataset, #sampler=sampler_test, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,)
    
    """Set up model for compute representation"""
    model = STTFEncoder(dim_in=args.dim_in,
                        num_classes=2, 
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
                        norm_layer=args.norm_layer, 
                        protocol="compute_representations")
    
    checkpoint_model = torch.load(args.model_path, map_location=device, weights_only=False)["model"]
    print("Load pre-trained model from: %s" % args.model_path)

    interpolate_temp_embed(model, checkpoint_model)

    # load pre-trained model
    model.load_state_dict(checkpoint_model, strict=False)

    compute_representations(model, loader_test, device, args)
    
    