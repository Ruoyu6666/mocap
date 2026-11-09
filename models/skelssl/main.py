"""
Span-masked, decoder-free (light predictor) SwAV + latent-regression SSL for skeleton sequences.

Input  : (B, T, 36, 2)  -> tokens (B, T/2, 12, D) -> spans of 6 time steps (12 frames), S = T/12
         36 joints = 3 mice x 12 keypoints; 12 groups = 3 mice x 4 body parts (3 joints each)

Student: block-mask whole spans, encode visible spans only (local attention inside span -> pool ->
         global attention over visible spans), light predictor with learnable mask tokens.
Teacher: EMA copy, sees ONE time step per span (random offset), no mask.
Loss   : SwAV (predictor output vs Sinkhorn codes of teacher span token) + smooth-L1 regression
         to the LayerNorm'ed teacher fine tokens, both on masked spans only.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SkeleSSL, spatial_aug, FRAMES_PER_SPAN, STEPS_PER_SPAN
from monitor import collapse_stats
import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from datasets.mabe_mice import MABeMouseDataset


# ----------------------------------------------------------------------------- training
def get_args(argv=None):
    p = argparse.ArgumentParser(description="Span-masked skeleton SSL (SwAV + latent regression)")
    g = p.add_argument_group("data")
    g.add_argument("--data", default="/home/rguo_hpc/myfolder/data/mabe_mice/mouse_triplet_train.npy",
                   help="MABe mouse triplet train .npy")
    g.add_argument("--seq-len", type=int, default=60, help="frames per crop, multiple of 12")
    g.add_argument("--sliding-window", type=int, default=10, help="stride between crops, in frames")
    g.add_argument("--no-fill-holes", action="store_true", help="keep missing keypoints as (0, 0)")
    g.add_argument("--workers", type=int, default=0)

    g = p.add_argument_group("model")
    g.add_argument("--dim", type=int, default=256)
    g.add_argument("--heads", type=int, default=8)
    g.add_argument("--depth1", type=int, default=2, help="local (within-span) blocks")
    g.add_argument("--depth2", type=int, default=4, help="global (across-span) blocks")
    g.add_argument("--pred-depth", type=int, default=2)
    g.add_argument("--pred-dim", type=int, default=128)
    g.add_argument("--rope-base", type=float, default=10000.0)
    g.add_argument("--n-groups", type=int, default=12)
    g.add_argument("--joints-per-group", type=int, default=3)
    g.add_argument("--patch-t", type=int, default=2)

    g = p.add_argument_group("masking / loss")
    g.add_argument("--mask-ratio", type=float, default=0.6)
    g.add_argument("--block-min", type=int, default=2)
    g.add_argument("--block-max", type=int, default=5)
    g.add_argument("--n-protos", type=int, default=256)
    g.add_argument("--proj-dim", type=int, default=128)
    g.add_argument("--student-temp", type=float, default=0.1)
    g.add_argument("--sinkhorn-eps", type=float, default=0.05)
    g.add_argument("--sinkhorn-iters", type=int, default=3)
    g.add_argument("--lambda-reg", type=float, default=1.0)

    g = p.add_argument_group("augmentation")
    g.add_argument("--no-rot", action="store_true")
    g.add_argument("--no-flip", action="store_true")
    g.add_argument("--noise", type=float, default=0.01)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--clip", type=float, default=3.0)
    g.add_argument("--ema-start", type=float, default=0.996)
    g.add_argument("--amp", action="store_true", help="bf16 autocast")
    
    g = p.add_argument_group("misc")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--save-dir", default="runs/default")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    assert a.seq_len % FRAMES_PER_SPAN == 0, "seq-len must be a multiple of 12"
    assert a.n_groups * a.joints_per_group == 36 and a.patch_t * STEPS_PER_SPAN == FRAMES_PER_SPAN
    return a


def main(argv=None):
    a = get_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.save_dir, exist_ok=True)
    # dataset-side augmentations stay off: spatial_aug below makes the two views on normalised coordinates
    ds = MABeMouseDataset(mode="train", path_to_data_dir=a.data, num_frames=a.seq_len,
                          sliding_window=a.sliding_window, interp_holes=not a.no_fill_holes)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size,
                                     shuffle=True, drop_last=True, num_workers=a.workers)
    model = SkeleSSL(a).to(a.device)
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(
        [{"params": [p for _, p in params if p.ndim > 1], "weight_decay": a.wd},
         {"params": [p for _, p in params if p.ndim <= 1], "weight_decay": 0.0}], lr=a.lr)
    total, step, t0 = a.epochs * len(dl), 0, time.time()
    dtype_dev = "cuda" if "cuda" in a.device else "cpu"

    for ep in range(a.epochs):
        for x, _ in dl:
            lr = a.lr * (step + 1) / a.warmup_steps if step < a.warmup_steps else \
                a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup_steps) / max(1, total - a.warmup_steps)))
            for gp in opt.param_groups:
                gp["lr"] = lr
            mom = 1 - (1 - a.ema_start) * (math.cos(math.pi * step / total) + 1) / 2
            x = x.to(a.device)
            xa = spatial_aug(x, not a.no_rot, not a.no_flip, a.noise)
            xb = spatial_aug(x, not a.no_rot, not a.no_flip, a.noise)
            with torch.autocast(dtype_dev, dtype=torch.bfloat16, enabled=a.amp):
                o = model(xa, xb)
            opt.zero_grad(set_to_none=True)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in params], a.clip)
            opt.step()
            model.update_teacher(mom)
            step += 1
            if step % a.log_every == 0 or step == 1:
                st = collapse_stats(o["t_scores"], o["s_scores"], o["q"], o["t_feat"])
                print(f"ep {ep} step {step}/{total} loss {o['loss'].item():.4f} swav {o['swav'].item():.4f} "
                      f"reg {o['reg'].item():.4f} lr {lr:.2e} ema {mom:.5f} "
                      + " ".join(f"{k} {v:.3f}" for k, v in st.items()) + f" | {time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep},
                   os.path.join(a.save_dir, "last.pt"))


if __name__ == "__main__":
    main()