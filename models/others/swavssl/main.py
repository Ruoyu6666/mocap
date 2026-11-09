import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SparseSSL

# ----------------------------------------------------------------------------- args
def get_parser():
    p = argparse.ArgumentParser(description="Sparse-view cross-time SSL (SwAV)")
    p.add_argument("--T", type=int, default=900, help="frames")
    p.add_argument("--n_joints", type=int, default=12)
    p.add_argument("--n_groups", type=int, default=4, help="joints 0-2 -> group 0, 3-5 -> group 1, ... (check your joint order)")
    p.add_argument("--patch_t", type=int, default=2, help="frames per token")
    p.add_argument("--span", type=int, default=6, help="time steps (tokens) per span")

    p.add_argument("--mask_ratio", type=float, default=0.8, help="fraction of the span*n_groups cells hidden from the student" )
    p.add_argument("--n_vis", type=int, default=5, help="visible student tokens per span, out of (span-1)*n_groups = 20")
    p.add_argument("--seg_spans", type=int, default=5, help="spans per global segment -> 30 tokens = 60 frames = 2 s")
    p.add_argument("--n_block_seg", type=int, default=3, help="segments per sequence with NO visible student tokens (0 disables)")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--pred_dim", type=int, default=128)
    p.add_argument("--pred_depth", type=int, default=2)
    p.add_argument("--pred_heads", type=int, default=4)
    p.add_argument("--num_prototype_local", type=int, default=128)
    p.add_argument("--num_prototype", type=int, default=128)


    p.add_argument("--proj_hidden", type=int, default=1024)
    p.add_argument("--proj_out", type=int, default=128)
    p.add_argument("--tau_s", type=float, default=0.1)
    p.add_argument("--sk_eps", type=float, default=0.05)
    p.add_argument("--sk_iters", type=int, default=3)

    p.add_argument("--lambda_local", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, default=1.0)
    p.add_argument("--lambda_glob", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--rope_ratio", type=float, default=0.5, help="fraction of each head's dims rotated by RoPE (1.0 = all)")
    p.add_argument("--seed", type=int, default=0)
    return p



if __name__ == "__main__":
    cfg = get_parser().parse_args()
    torch.manual_seed(cfg.seed)
    model = SparseSSL(**{k: v for k, v in vars(cfg).items() if k != "seed"})
    x = torch.randn(4, cfg.T, cfg.n_joints, 2)

    # 1) the teacher's time step is never among the student's visible tokens of its span
    t_teacher, flat = model.sample_views(4, x.device)
    t_s = flat // cfg.n_groups
    span_s = t_s // cfg.span
    assert (torch.gather(t_teacher, 1, span_s) != t_s).all(), "teacher step leaked into student"
    print("visible student tokens:", flat.shape[1], "| teacher tokens:", t_teacher.shape[1] * cfg.n_groups)

    # 2) forward / backward / EMA
    loss, logs = model(x)
    loss.backward()
    model.update_teacher(0.996)
    print("loss", float(loss), {k: round(float(v), 4) for k, v in logs.items()})
    assert all(p.grad is None for p in list(model.tok_t.parameters()) + list(model.enc_t.parameters()))

    # 3) dense encoding, incl. a longer sequence (length extrapolation path)
    print("dense", cfg.T, ":", tuple(model.encode_dense(x).shape))
    print("dense", 4 * cfg.T, ":", tuple(model.encode_dense(torch.randn(1, 4 * cfg.T, cfg.n_joints, 2)).shape))
    print("OK")
