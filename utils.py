import os
import time
import numpy as np
import argparse
import pickle
import torch
from torch.utils.data import DataLoader


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()


def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_dirs(save_dir):
    existing_versions = os.listdir(save_dir)
        
    if len(existing_versions) > 0:
        max_version = int(existing_versions[0].split("_")[-1])
        for v in existing_versions:
            ver = int(v.split("_")[-1])
            if ver > max_version:
                    max_version = ver
        version = int(max_version) + 1
    else:
        version = 0
    # os.makedirs(directory, exist_ok=True)
    return f"{save_dir}/exp_{version}"



def save_checkpoint(model, optimizer, epoch, args):
    """Saves the model checkpoint"""
    SAVE_CHECKPOINT_PATH = args.save_dir + 'checkpoints'
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args
    }
    torch.save(checkpoint, SAVE_CHECKPOINT_PATH + '/vqvae_checkpoint'+ str(epoch) +'.pth')



def save_model(model, optimizer, args):
    """Saves the final model after training"""
    SAVE_MODEL_PATH = args.save_dir + 'models'
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    model = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'args': args
        }
    torch.save(model, SAVE_MODEL_PATH + '/vqvae_model.pth')    



def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']
    return model, optimizer, epoch, args



def save_results(results, args):
    SAVE_RESULT_PATH = args.save_dir + 'results'
    os.makedirs(SAVE_RESULT_PATH, exist_ok=True)
    # to change
    with open(SAVE_RESULT_PATH + '/vqvae_results_' + '.pkl', "wb") as f:
        pickle.dump(results, f)
