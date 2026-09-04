import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")
from swav.finetune.layers import ProjectionHead, PrototypeLayer
from swav.finetune.utils import sinkhorn, swav_loss


# 1. Dataset over precomputed paired embeddings
class PairedEmbeddingDataset(Dataset):
    def __init__(self, view_a: np.array, view_b: np.array):
        self.z_a = torch.from_numpy(view_a).float()
        self.z_b = torch.from_numpy(view_b).float()
        assert self.z_a.shape == self.z_b.shape, (f"view shapes must match: {self.z_a.shape} vs {self.z_b.shape}")

    def __len__(self):
        return self.z_a.shape[0]

    def __getitem__(self, idx):
        return self.z_a[idx], self.z_b[idx]



# 2. Training loop (prototypes only, no encoder)
def train_prototypes(view_a: np.array, view_b: np.array, num_prototypes: int, 
                     batch_size: int, num_epochs: int, lr: float, weight_decay: float, 
                     freeze_prototypes_epoch0: bool = True, gmm_means: np.ndarray = None,
                     device: str = "cuda", log_every: int = 100, 
                     projection_hidden_dim: int = 192, projection_out_dim: int = None,):
    
    dataset = PairedEmbeddingDataset(view_a, view_b)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    embed_dim = dataset.z_a.shape[1]
    
    projection_head = None
    if projection_out_dim is not None:
        projection_head = ProjectionHead(embed_dim, projection_hidden_dim, projection_out_dim).to(device)
    
    prototypes = PrototypeLayer(embed_dim, num_prototypes).to(device)
    
    if gmm_means is not None: # warm-start from your existing GMM cluster centers, if you have them
        # NOTE: gmm_means must be in the same space the prototypes operate on — i.e. in projection_out_dim space 
        # ONLY use this warm-start when projection_out_dim=None and your GMM was fit on the raw frozen embeddings
        centers = torch.as_tensor(gmm_means, dtype=torch.float32)
        prototypes.init_from_centers(F.normalize(centers, dim=1, p=2))

    trainable_params = list(prototypes.parameters())
    if projection_head is not None:
        trainable_params += list(projection_head.parameters())

    optimizer = torch.optim.AdamW(prototypes.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (z_a, z_b) in enumerate(loader):
            z_a, z_b = z_a.to(device), z_b.to(device)
            if projection_head is not None:
                p_a = projection_head(z_a)
                p_b = projection_head(z_b)
            else:
                p_a, p_b = z_a, z_b
 
            p_a = F.normalize(p_a, dim=1, p=2)
            p_b = F.normalize(p_b, dim=1, p=2)
            scores_a = prototypes(z_a)
            scores_b = prototypes(z_b)

            loss = swav_loss(scores_a, scores_b)
            optimizer.zero_grad()
            loss.backward()
            if freeze_prototypes_epoch0 and epoch == 0:
                for p in prototypes.parameters():
                    p.grad = None
            optimizer.step()
            prototypes.normalize_prototypes()
            running_loss += loss.item()
            if step % log_every == 0:
                print(f"epoch {epoch:3d}  step {step:4d}  loss {loss.item():.4f}")
        print(f"epoch {epoch:3d} done, avg loss {running_loss / len(loader):.4f}")

    return prototypes, projection_head, dataset




# --------------------------------------------------------------------------
# 3. Compute new representations using the trained prototypes
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_new_representations(prototypes: PrototypeLayer, embeddings: torch.Tensor, device: str,
                                projection_head: nn.Module = None, which: str = "cluster",
                                temperature: float = 0.1, batch_size: int = 1024,):
    """
    embeddings: (N, D) raw frame embeddings (NOT yet L2-normalized)
    which:
      "cluster"    - (N, K) softmax-over-prototypes soft assignment. Always available. This is the only new signal you get 
                    if projection_head is None (prototypes-only training).
      "projection" - (N, D') the projection head's output (L2-normalized), i.e. an actual new learned embedding. 
                    Requires projection_head is not None.
    """
    prototypes = prototypes.to(device)
    prototypes.eval()
    if projection_head is not None:
        projection_head = projection_head.to(device)
        projection_head.eval()

    out = []
    for i in range(0, embeddings.shape[0], batch_size):
        chunk = embeddings[i:i + batch_size].to(device)
        p = projection_head(chunk) if projection_head is not None else chunk
        z = F.normalize(p, dim=1, p=2)
        if which == "projection":
            out.append(z.cpu())
        elif which == "cluster":
            scores = prototypes(z)
            probs = F.softmax(scores / temperature, dim=1)
            out.append(probs.cpu())
    
    return torch.cat(out, dim=0)