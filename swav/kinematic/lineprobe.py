import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



# 1. Dataset
class KinematicDataset(Dataset):
    """Wraps precomputed embeddings (N, L, C) and kinematic targets (N, L, 20)."""
    def __init__(self, embed_path, target_path):
        self.embeds = np.load(embed_path).astype(np.float32)   # (N, L, C)
        self.targets = np.load(target_path).astype(np.float32) # (N, L, 20)
        assert self.embeds.shape[:2] == self.targets.shape[:2], f"shape mismatch: {self.embeds.shape} vs {self.targets.shape}"

    def __len__(self):
        return self.embeds.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.embeds[idx]), torch.from_numpy(self.targets[idx]))    # (L,20)
                


# 2. Model: frozen-embedding adapter + regression head
class KinematicRegressionHead(nn.Module):
    def __init__(self, embed_dim, num_features=20, hidden_dim=128, dropout=0.1, per_dim_weight=False):
        """
        per_dim_weight: False -> single learnable scalar weight shared across all embed dims
                        True  -> one learnable weight per embedding dimension
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, embed_dim),)

        weight_shape = (embed_dim,) if per_dim_weight else (1,)
        # init at 0.0 -> sigmoid(0.0) = 0.5, i.e. start as an even mix
        self.raw_weight = nn.Parameter(torch.zeros(weight_shape))
        self.regressor = nn.Linear(embed_dim, num_features)

    def forward(self, z):
        w = torch.sigmoid(self.raw_weight)          # constrained to (0, 1)
        z_tuned = w * z + (1 - w) * self.net(z)     # (B, L, C)
        pred = self.regressor(z_tuned)
        return z_tuned, pred


# 3. Loss
class KinematicLoss(nn.Module):
    def __init__(self, num_features, feature_mean, feature_std, learn_weights=True):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.learn_weights = learn_weights
        if learn_weights:
            self.log_var = nn.Parameter(torch.zeros(num_features))

    def normalize(self, target):
        return (target - self.feature_mean) / (self.feature_std + 1e-6)

    def forward(self, pred, target):
        target_norm = self.normalize(target)                          # (B, L, F)
        per_elem = F.smooth_l1_loss(pred, target_norm, reduction="none")  # (B, L, F)
        per_feature_mean = per_elem.reshape(-1, per_elem.shape[-1]).mean(dim=0)  # (F,)

        if self.learn_weights:
            log_var = torch.clamp(self.log_var, -5.0, 5.0)
            precision = torch.exp(-log_var)
            loss = (precision * per_feature_mean + log_var).sum()
        else:
            loss = per_feature_mean.sum()
        
        return loss, per_feature_mean.detach()



# 4. Compute normalization stats from the TRAIN set only
def compute_feature_stats(target_path):
    targets = np.load(target_path).astype(np.float32)   # (N, L, 20)
    flat = targets.reshape(-1, targets.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))



# 5. Train / validate loop
def train(train_embed_path, train_target_path,val_embed_path, val_target_path,
          embed_dim=128, num_features=20,
          batch_size=64, epochs=50, lr=1e-4, weight_decay=1e-4,
          patience=8, device="cuda" if torch.cuda.is_available() else "cpu",
          ckpt_path="best_kinematic_model.pt",):
    # --- data ---
    train_ds = KinematicDataset(train_embed_path, train_target_path )
    val_ds   = KinematicDataset(val_embed_path, val_target_path,)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    feat_mean, feat_std = compute_feature_stats(train_target_path)

    # --- model / loss / optim ---
    model = KinematicRegressionHead(embed_dim, num_features).to(device)
    criterion = KinematicLoss(num_features, feat_mean, feat_std, learn_weights=False).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=weight_decay,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(epochs):
        # ---- train ----
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for z, target in train_loader:
            z, target = z.to(device), target.to(device)
            _, pred = model(z)
            loss, per_feat = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / n_batches

        # ---- validate ----
        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        per_feat_sum = torch.zeros(num_features, device=device)
        with torch.no_grad():
            for z, target, in val_loader:
                z, target, = z.to(device), target.to(device)
                _, pred = model(z)
                loss, per_feat = criterion(pred, target)
                val_loss_sum += loss.item()
                per_feat_sum += per_feat
                n_val_batches += 1
        val_loss = val_loss_sum / n_val_batches
        per_feat_avg = (per_feat_sum / n_val_batches).cpu().numpy()

        print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | " f"per-feature (first 5) {np.round(per_feat_avg[:5], 3)}")

        # ---- early stopping / checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "criterion_state": criterion.state_dict(),
                        "feat_mean": feat_mean,
                        "feat_std": feat_std,
                        "embed_dim": embed_dim,
                        "num_features": num_features,}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    print(f"best val loss: {best_val_loss:.4f}, checkpoint saved to {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# 6. Run inference: produce the TUNED embeddings and save them
# ---------------------------------------------------------------------------
@torch.no_grad()
def export_tuned_features(ckpt_path, embed_path, out_path, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu",):

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = KinematicRegressionHead(ckpt["embed_dim"], ckpt["num_features"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embeds = np.load(embed_path).astype(np.float32)  # (N, L, C)
    N = embeds.shape[0]
    out = np.zeros_like(embeds)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        z = torch.from_numpy(embeds[start:end]).to(device)
        z_tuned,_ = model(z)
        out[start:end] = z_tuned.cpu().numpy()

    np.save(out_path, out)
    print(f"saved tuned features {out.shape} -> {out_path}")
    return out




# ---------------------------------------------------------------------------
# 7. Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ckpt = train(
        train_embed_path="./features/mae_feats_pca_tr.npy",   # (N_train, L, C)
        train_target_path="./features/kinematic_feats_tr.npy",   # (N_train, L, 20)
        val_embed_path="./features/mae_feats_pca_val.npy",        # (N_val, L, C)
        val_target_path="./features/kinematic_feats_val.npy",    # (N_val, L, 20)
        embed_dim=128,
        num_features=20,
        batch_size=32,
        epochs=30,
        lr=5e-4,)
    
    # Export tuned features for both splits (or your full dataset)
    export_tuned_features(ckpt, "./features/mae_feats_pca_tr.npy", "train_embeddings_tuned.npy")
    export_tuned_features(ckpt, "./features/mae_feats_pca_val.npy", "val_embeddings_tuned.npy")