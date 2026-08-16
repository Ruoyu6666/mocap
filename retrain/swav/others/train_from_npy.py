"""
Train a SwAV prototype head directly on precomputed frame embeddings.

Use this when you've already run your (frozen) SkeletonMAE encoder and saved two views' worth of frame-level embeddings to disk as .npy files:
view_a.npy : (N, D)  e.g. embeddings of frame t
view_b.npy : (N, D)  e.g. embeddings of a temporally nearby frame t' (same N frames, paired row-for-row with view_a)

No encoder forward pass happens here — only the PrototypeLayer (C in R^{D x K}) is trained. This reuses PrototypeLayer / sinkhorn / swav_loss 
from swav_skeleton_finetune.py, so keep that file in the same directory (or adjust the import).

Output:
    trained_prototypes.pt   - the learned prototype weight matrix
    new_representations.npy - (N, K) soft cluster/prototype assignment per frame, computed from view_a 
                              (see note below on why view_a and not an average of both views)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import sys
sys.path.append("/home/rguo_hpc/myfolder/mocap")

from retrain.swav.layers import ProjectionHead, PrototypeLayer
from retrain.swav.utils import sinkhorn, swav_loss


# --------------------------------------------------------------------------
# 1. Dataset over precomputed paired embeddings
# --------------------------------------------------------------------------
class PairedEmbeddingDataset(Dataset):
    def __init__(self, view_a_path: str, view_b_path: str):
        self.z_a = np.load(view_a_path)
        self.z_b = np.load(view_b_path)
        assert self.z_a.shape == self.z_b.shape, (f"view shapes must match: {self.z_a.shape} vs {self.z_b.shape}")

        self.z_a = torch.from_numpy(self.z_a).float()
        self.z_b = torch.from_numpy(self.z_b).float()

    def __len__(self):
        return self.z_a.shape[0]

    def __getitem__(self, idx):
        return self.z_a[idx], self.z_b[idx]




# --------------------------------------------------------------------------
# 2. Training loop (prototypes only, no encoder)
# --------------------------------------------------------------------------

def train_prototypes(view_a_path: str, view_b_path: str,
                    num_prototypes: int = 60,
                    batch_size: int = 1024,
                    num_epochs: int = 50,
                    lr: float = 1e-3,
                    weight_decay: float = 1e-4,
                    freeze_prototypes_epoch0: bool = True,
                    gmm_means: np.ndarray = None,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    log_every: int = 20,):
    
    dataset = PairedEmbeddingDataset(view_a_path, view_b_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    embed_dim = dataset.z_a.shape[1]
    prototypes = PrototypeLayer(embed_dim, num_prototypes).to(device)

    if gmm_means is not None:
        # warm-start from your existing GMM cluster centers, if you have them
        centers = torch.as_tensor(gmm_means, dtype=torch.float32)
        prototypes.init_from_centers(F.normalize(centers, dim=1, p=2))

    optimizer = torch.optim.AdamW(prototypes.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (z_a, z_b) in enumerate(loader):
            z_a = F.normalize(z_a.to(device), dim=1, p=2)
            z_b = F.normalize(z_b.to(device), dim=1, p=2)

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

    return prototypes, dataset


# --------------------------------------------------------------------------
# 3. Compute new representations using the trained prototypes
# --------------------------------------------------------------------------

@torch.no_grad()
def compute_new_representations(prototypes: PrototypeLayer,
                                embeddings: torch.Tensor,
                                device: str,
                                temperature: float = 0.1,
                                batch_size: int = 4096,):
    """
    embeddings: (N, D) raw frame embeddings (NOT yet L2-normalized)
    Returns: (N, K) soft prototype assignment per frame — softmax over prototype similarity, 
    same transform used inside the SwAV loss but applied deterministically (no Sinkhorn equipartition constraint 
    at inference time, since that constraint only makes sense over a batch during training).
    """
    prototypes = prototypes.to(device)
    prototypes.eval()
    out = []
    for i in range(0, embeddings.shape[0], batch_size):
        chunk = embeddings[i:i + batch_size].to(device)
        z = F.normalize(chunk, dim=1, p=2)
        scores = prototypes(z)
        probs = F.softmax(scores / temperature, dim=1)
        out.append(probs.cpu())
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------
# 4. End-to-end script
# --------------------------------------------------------------------------

if __name__ == "__main__":
    VIEW_A_PATH = "view_a.npy"   # <-- adjust paths
    VIEW_B_PATH = "view_b.npy"
    NUM_PROTOTYPES = 60          # <-- match your expected number of behaviors
    NUM_EPOCHS = 50

    prototypes, dataset = train_prototypes(VIEW_A_PATH, VIEW_B_PATH, num_prototypes=NUM_PROTOTYPES, num_epochs=NUM_EPOCHS,)

    torch.save(prototypes.state_dict(), "trained_prototypes.pt")
    print("Saved trained_prototypes.pt")

    # Compute the new K-dim representation for every frame in view_a. (Using view_a as "the" per-frame representation 
    # since view_b was only a temporal-pairing signal for training, not a separate population of frames you necessarily 
    # want represented — adjust if your two files actually cover different/non-overlapping frames you want stacked.)
    new_repr = compute_new_representations(prototypes, dataset.z_a)
    np.save("new_representations.npy", new_repr.numpy())
    print(f"Saved new_representations.npy with shape {tuple(new_repr.shape)}")

    # If you'd rather keep the original D-dim embedding AND add the cluster representation (rather than replace it), concatenate instead:
    # combined = torch.cat([F.normalize(dataset.z_a, dim=1), new_repr], dim=1)
    # np.save("combined_representations.npy", combined.numpy())