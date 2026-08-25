import math
import torch
import torch.nn as nn
import torch.nn.functional as F




def temporal_shift_view(sequence: torch.Tensor, center_idx: torch.Tensor, max_shift: int = 2):
    """
    Optional: instead of (or in addition to) augmenting the same frame twice, pull the second view from a nearby frame 
    in the same sequence. Encourages the representation to be smooth over short time windows.

    sequence: (B, T, J, C) — a window of frames per sample
    center_idx: (B,) index of the "anchor" frame within each window
    Returns: (B, J, C) frames at center_idx + random offset in [-max_shift, max_shift]
    """
    B, T, J, C = sequence.shape
    offset = torch.randint(-max_shift, max_shift + 1, (B,), device=sequence.device)
    idx = (center_idx + offset).clamp(0, T - 1)
    return sequence[torch.arange(B), idx]  # (B, J, C)




# --------------------------------------------------------------------------
# 6. Example training loop
# --------------------------------------------------------------------------
def train_one_epoch(model: SwAVSkeletonModel, dataloader, optimizer, augment: SkeletonAugment, device: str = "cuda",
                    freeze_prototypes_epoch: bool = False, log_every: int = 50,):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        # batch: (B, J, C) raw frame skeleton coords — adapt to your loader
        x = batch.to(device)

        x_a = augment(x)
        x_b = augment(x)

        z_a, scores_a = model(x_a)
        z_b, scores_b = model(x_b)

        loss = swav_loss(scores_a, scores_b)

        optimizer.zero_grad()
        loss.backward()

        if freeze_prototypes_epoch:
            # standard SwAV trick: freeze prototype gradients for epoch 0
            for p in model.prototypes.parameters():
                p.grad = None

        optimizer.step()
        model.prototypes.normalize_prototypes()  # keep C on unit sphere

        running_loss += loss.item()
        if step % log_every == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    return running_loss / max(1, len(dataloader))


# --------------------------------------------------------------------------
# Usage sketch
# --------------------------------------------------------------------------
"""
encoder = load_pretrained_skeletonmae_encoder()   # your existing encoder
embed_dim = 128                                    # match your MAE output dim
num_prototypes = 60                                # e.g. match your GMM component count

model = SwAVSkeletonModel(encoder, embed_dim, num_prototypes).to(device)

# optional: warm-start prototypes from your existing GMM clustering
# init_prototypes_from_gmm(model, gmm.means_)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
augment = SkeletonAugment(joint_dropout_p=0.1, jitter_std=0.01, rot_deg=15.0)

for epoch in range(num_epochs):
    freeze = (epoch == 0)  # SwAV convention: freeze prototype grads epoch 0
    avg_loss = train_one_epoch(model, dataloader, optimizer, augment, device=device, freeze_prototypes_epoch=freeze)
    print(f"epoch {epoch}: avg loss {avg_loss:.4f}")
"""