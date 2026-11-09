"""
Collapse monitoring for the span-masked skeleton SSL model.

Used in two ways:
  * imported by skel_ssl.py to log stats during training (collapse_stats)
  * run as a CLI on a checkpoint + data file (python monitor.py --ckpt ... --data ...)
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from model import PoseCrops, SkeleSSL

def _entropy_exp(p):
    p = p / p.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum())


@torch.no_grad()
def eff_rank(f, max_n=4096):
    """exp(entropy of normalised singular values). Low value => dimensional collapse."""
    f = f.float()
    if f.size(0) > max_n:
        f = f[torch.randperm(f.size(0), device=f.device)[:max_n]]
    f = f - f.mean(0, keepdim=True)
    s = torch.linalg.svdvals(f)
    return _entropy_exp(s).item()


@torch.no_grad()
def feat_spread(f):
    """Std of l2-normalised features, scaled so isotropic features give ~1 and a constant gives 0."""
    f = F.normalize(f.float(), dim=-1)
    return (f.std(0).mean() * math.sqrt(f.size(-1))).item()


@torch.no_grad()
def collapse_stats(t_scores, s_scores=None, q=None, feats=None):
    """
    t_scores: (N, K) RAW teacher prototype scores (before Sinkhorn; Sinkhorn forces equal usage,
              so usage must be measured on the raw scores).
    s_scores: (N, K) student scores (optional) -> argmax agreement with the Sinkhorn codes q.
    feats:    (N, D) features -> spread and effective rank.
    """
    K = t_scores.size(-1)
    hist = torch.bincount(t_scores.argmax(-1), minlength=K).float()
    out = {
        "proto_ppl": _entropy_exp(hist).item(),  # K = uniform usage, 1 = single prototype
        "proto_used_frac": (hist > 0).float().mean().item(),
    }
    if s_scores is not None and q is not None:
        out["agree"] = (s_scores.argmax(-1) == q.argmax(-1)).float().mean().item()
    if feats is not None:
        out["feat_spread"] = feat_spread(feats)
        out["eff_rank"] = eff_rank(feats)
    return out


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Collapse monitoring on a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="", help=".npy of shape (N, L, 12, 2); empty = synthetic")
    p.add_argument("--seq-len", type=int, default=0, help="0 = use the training seq-len")
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--teacher", action="store_true", help="use the EMA teacher instead of the student")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):

    a = get_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu")
    ta = argparse.Namespace(**ck["args"])
    model = SkeleSSL(ta)
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    T = a.seq_len or ta.seq_len
    ds = PoseCrops(a.data, T, n_synth=a.n_batches * a.batch_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True)

    fine_l, span_l, score_l = [], [], []
    with torch.no_grad():
        for i, x in enumerate(dl):
            if i >= a.n_batches:
                break
            fine, span = model.extract(x.to(a.device), fuse="mean", teacher=a.teacher)
            head = model.t_head if a.teacher else model.head
            z = F.normalize(head(span), dim=-1)
            W = F.normalize(model.protos.weight, dim=1)
            fine_l.append(fine.flatten(0, 1).cpu())
            span_l.append(span.flatten(0, 1).cpu())
            score_l.append((z @ W.t()).flatten(0, 1).cpu())
    fine, span, scores = torch.cat(fine_l), torch.cat(span_l), torch.cat(score_l)

    print(f"frames-tokens: {fine.shape[0]}  span tokens: {span.shape[0]}")
    print("span level :", collapse_stats(scores, feats=span))
    print("fine level :", {"feat_spread": feat_spread(fine), "eff_rank": eff_rank(fine)})
    print("Healthy signs: proto_ppl well above 1 (ideally a large fraction of K), feat_spread >> 0,"
          " eff_rank a decent fraction of the feature dim.")


if __name__ == "__main__":
    main()
