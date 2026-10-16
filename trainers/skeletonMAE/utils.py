import math



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