import os
import math
import time
import argparse
import torch



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



def adjust_learning_rate(optimizer, epoch, args):
    """Linear warmup then half-cosine decay. `epoch` may be fractional (per-iteration schedule)."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        decay_epochs = max(args.epochs - args.warmup_epochs, 1e-8)  # guards epochs == warmup_epochs
        progress = min((epoch - args.warmup_epochs) / decay_epochs, 1.0)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group: 
            param_group["lr"] = lr * param_group["lr_scale"] 
        else: 
            param_group["lr"] = lr
    return lr



def build_param_groups(model, weight_decay):
    """No weight decay on biases, norm params, 1-D params, tokens and positional embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},]


def save_model(model, optimizer, args):
    """Saves the final model after training"""
    os.makedirs(args.save_dir, exist_ok=True)
    model = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'args': args
        }
    torch.save(model, args.save_dir + 'model.pth') 