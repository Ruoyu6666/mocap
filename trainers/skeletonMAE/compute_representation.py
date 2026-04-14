"""
mouse_names = {
    "CP1A": ['M1', 'M14', 'M15', 'M19'], 
    "CP1B": ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'], 
    "INH1": ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'], 
    "INH2": ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12'], 
    "MOS1aD": ['M4', 'M5', 'M6', 'M8', 'M9', 'M10']}
"""

# k fold validation based on mouse
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
               "valid": ["M4"]}
}
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
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/mocap/data/mocap/data_CLB.pkl')
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
    
    # model path for computing representation
    parser.add_argument("--model_path", type=str, default="/home/rguo_hpc/myfolder/mocap/outputs/checkpoints/mae_checkpoint_epoch_25.pth")
    
    """Type of job"""
    parser.add_argument("--if_val", type=str2bool, default=False) # whether to compute representations for validation set (if False, compute for training set)
    parser.add_argument("--job", type=str, default="compute_representations", choices=["pretrain", "compute_representations","linprobe", "finetune"])
    parser.add_argument("--fast_inference",default=True)
    return parser.parse_args()




# fast inference to compute representations for all sequences in the dataset and save as .npy file
def compute_representations(model,data_loader, device, args, num_sequences):
    os.makedirs(args.save_dir + '/representations', exist_ok=True)
    model = model.to(device)
    model.eval()
    all_representations = []
    
    if args.fast_inference:
        with torch.no_grad():
            for i, (x, _)  in enumerate(data_loader):
                x = x.to(device)
                latent = model(x)      # (N, T, C)
                all_representations.append(torch.squeeze(latent).cpu().numpy())
                if (i + 1) % args.log_interval == 0:
                    print(f"Processed {i+1}/{len(data_loader)} batches.")

        all_representations = np.concatenate(all_representations, axis=0)
        all_representations = all_representations.reshape(num_sequences, -1, args.dim_feat) # (N, T, C)
    #### to be completed: non-fast inference that averages representations for overlapping subsequences to get representation for the whole sequence
    else:
        full_len = data_loader.dataset.seq_keypoints.shape[1]
        starts = list(range(0, full_len - args.num_frames + 1, args.sliding_window))
        repr_sum = torch.zeros(num_sequences, int(full_len/3), args.dim_feat)
        count_sum = torch.zeros(num_sequences, int(full_len/3), 1)
        
        
        for i, (x, _)  in enumerate(tqdm(data_loader)): # i, index of the batch; x, batch of subsequences; keypoints_id, list of tuples (seq_id, start_idx) for each subsequence in the batch
            keypoints_id = data_loader.dataset.keypoints_ids[i*args.batch_size:(i+1)*args.batch_size] # list of tuples (seq_id, start_idx) for each subsequence in the batch
            x = x.to(device)
            latent = model(x)      # (N, T, C)
            latent = torch.squeeze(latent).cpu().detach().numpy() # (N, T, C)

            for j in range(len(keypoints_id)):
                seq_id, start_idx = keypoints_id[j]
                repr_sum[seq_id, start_idx:start_idx+100] += torch.from_numpy(latent[j])
                count_sum[seq_id, start_idx:start_idx+100] += 1
        all_representations = repr_sum / count_sum # (N, T, C)
    
    if args.if_val:
        np.save(args.save_dir + '/representations/mae_'+ args.dataset +'_val.npy', all_representations)
    else:
        np.save(args.save_dir + '/representations/mae_'+ args.dataset +'_tr.npy', all_representations)




if __name__ == "__main__":

    timestamp = readable_timestamp()
    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.job == "compute_representations":
        """Set up data set and data loader"""
        if args.dataset == "mabe_mice":
            dataset = MABeMouseDataset(path_to_data_dir=args.path_to_data_dir,
                                        sampling_rate=args.sampling_rate,
                                        num_frames=args.num_frames, 
                                        sliding_window=args.num_frames-1,
                                        if_fill=args.fill_holes,
                                        # cache_path=args.cache_path, cache=False,
                                        augmentations=None,
                                        include_testdata=True,)
        elif args.dataset == "mocap":
            dataset = MocapDataset(mode = args.job,
                                path_to_data_dir = args.path_to_data_dir,
                                datasets = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"],
                                task = args.task , # FL2 or Tr
                                sampling_rate=args.sampling_rate,
                                num_frames=args.num_frames,
                                sliding_window=args.num_frames if args.fast_inference else args.sliding_window,
                                fill_holes=args.fill_holes,
                                augmentations=args.data_augment,
                                view_invariant = True, 
                                left_idx = 3,       # default left hip
                                right_idx = 8,       # default right hip
                                index_frame = 149, 
                                model = "SkeletonMAE",
                                split = fold_1, # whether to split dataset by mouse for train/val
                                if_val = args.if_val,)
            
            num_sequences = dataset.seq_keypoints.shape[0]

        loader = DataLoader(dataset, #sampler=sampler_test, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,)


        # Set up encoder-only model for compute representation
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

        compute_representations(model, loader, device, args, num_sequences)
    