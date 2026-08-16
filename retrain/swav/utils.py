import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 2. Sinkhorn-Knopp: turns raw scores into equipartitioned soft codes
# --------------------------------------------------------------------------
 
@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """
    scores: (B, K) similarity logits (z @ C), NOT yet softmaxed.
    Returns Q: (B, K) soft assignment, each row sums to 1, each column
    (approximately) equal mass -> prevents collapse to a few prototypes.
    """
    Q = torch.exp(scores / eps).T  # (K, B)
    B = Q.shape[1]
    K = Q.shape[0]
 
    # normalize Q to be a valid transport plan
    sum_Q = Q.sum()
    Q /= sum_Q
 
    for _ in range(n_iters):
        # normalize rows (each prototype gets equal total mass)
        row_sum = Q.sum(dim=1, keepdim=True)
        Q /= row_sum
        Q /= K
        # normalize columns (each sample sums to 1)
        col_sum = Q.sum(dim=0, keepdim=True)
        Q /= col_sum
        Q /= B
 
    Q *= B  # columns sum to 1 (soft assignment per sample)
    return Q.T  # (B, K)



# --------------------------------------------------------------------------
# 4. SwAV loss
# --------------------------------------------------------------------------
 
def swav_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 3,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    scores_a, scores_b: (B, K) raw prototype logits (z @ C) for two views.
    Swapped prediction: codes from view A predict view B's distribution and
    vice versa.
    """
    with torch.no_grad():
        q_a = sinkhorn(scores_a, eps=sinkhorn_eps, n_iters=sinkhorn_iters)
        q_b = sinkhorn(scores_b, eps=sinkhorn_eps, n_iters=sinkhorn_iters)
 
    p_a = F.log_softmax(scores_a / temperature, dim=1)
    p_b = F.log_softmax(scores_b / temperature, dim=1)
 
    loss = -0.5 * (
        (q_b * p_a).sum(dim=1).mean() + (q_a * p_b).sum(dim=1).mean()
    )
    return loss






# --------------------------------------------------------------------------
# 7. Initializing prototypes from your existing GMM / watershed centers
# --------------------------------------------------------------------------

def init_prototypes_from_gmm(model: SwAVSkeletonModel, gmm_means, embed_normalize=True):
    """
    gmm_means: numpy array or tensor (K, D) — means of your existing GMM fit
    on pretrained SkeletonMAE embeddings. K must equal model.prototypes'
    num_prototypes and D must equal embed_dim.
    """
    centers = torch.as_tensor(gmm_means, dtype=torch.float32)
    if embed_normalize:
        centers = F.normalize(centers, dim=1, p=2)
    model.prototypes.init_from_centers(centers)


def build_optimizer(model: SwAVSkeletonModel, lr: float = 1e-4, weight_decay: float = 1e-4):
    """
    Only passes parameters with requires_grad=True to the optimizer.
    Matters most in 'freeze' / 'finetune_last_n' modes — otherwise you'd
    waste memory on optimizer state (e.g. AdamW momentum buffers) for
    parameters that never get a gradient.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)