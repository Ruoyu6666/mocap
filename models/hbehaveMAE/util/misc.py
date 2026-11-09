# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
#
# Enhanced and modified by Stoffl et al.
#
# For more details on our work, please refer to:
# Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders
# Lucas Stoffl, Andy Bonnetto, Stéphane d'Ascoli, Alexander Mathis
# https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1
# --------------------------------------------------------

import argparse
import builtins
import datetime
import os
import time
from collections import OrderedDict, defaultdict, deque

import psutil
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr as pathmgr

# from torch._six import inf
from torch import inf

import models.hbehaveMAE.util.logging as logging
from models.hbehaveMAE.util.logging import master_print as print

logger = logging.get_logger(__name__)



def get_last_checkpoint(args):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = args.output_dir
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        print(d, name)
        return os.path.join(d, name)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if not args.resume:
        args.resume = get_last_checkpoint(args)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            with pathmgr.open(args.resume, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        
        if ("optimizer" in checkpoint and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)):
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")



def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def inflate(model_2d, model_3d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "patch_embed.proj.weight" in k:
            v3d = model_3d[k]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            state_dict_inflated[k] = v3d.clone()
        elif "pos_embed" in k:
            pos_embed_cls, pos_embed_spatial = torch.split(v2d, [1, 196], dim=1)
            state_dict_inflated["pos_embed_cls"] = pos_embed_cls.clone()
            state_dict_inflated["pos_embed"] = pos_embed_spatial.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def convert_checkpoint(model_2d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "head.projection.weight" in k:
            state_dict_inflated["head.weight"] = v2d.clone()
        elif "head.projection.bias" in k:
            state_dict_inflated["head.bias"] = v2d.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def parse_tuples(arg):
    try:
        return [tuple(map(int, item.split(","))) for item in arg.split(";")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid tuple format. Expected comma-separated numbers."
        )