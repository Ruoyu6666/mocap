import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



@torch.no_grad()
def compute_representations(model, dataloader, device, which: str = "projection", t_patch_size: int = 1,):
    
    assert which in ("projection", "raw", "cluster")
    model.eval()
    mae = model.mae

    dataset = dataloader.dataset
    num_sequences = dataset.num_sequences
    full_len = dataset.seq_keypoints.shape[1]
    T_tokens = full_len // t_patch_size
    count_sum = torch.zeros(num_sequences, T_tokens, 1)
    repr_sum = None  # allocated once we know D_out, from the first batch
    item_ptr = 0
    for i, (x, _)  in enumerate(tqdm(dataloader)):
        x = x.to(device)
        z_seq, _ = mae.forward_encoder_full(x)  # (N, L, D)
        B, latent_len, D = z_seq.shape # L = TP * VP
 
        if which == "raw":
            out = z_seq
        else:
            flat = z_seq.reshape(-1, D)
            p = model.projection_head(flat) if model.projection_head is not None else flat
            p = F.normalize(p, dim=1, p=2)

            if which == "projection":
                out = p.reshape(B, latent_len, -1)
            elif which == "cluster":
                scores = model.prototypes(p)
                probs = F.softmax(scores / 0.1, dim=1)
                out = probs.reshape(B, latent_len, -1)
        out = out.cpu()

        if repr_sum is None:
            D_out = out.shape[-1]
            repr_sum = torch.zeros(num_sequences, T_tokens, D_out)
        keypoints_id_batch = dataset.keypoints_ids[item_ptr:item_ptr + B]
        item_ptr += B
        for j, (seq_id, start_idx) in enumerate(keypoints_id_batch):
            start_token = int(start_idx / t_patch_size)
            end_token = start_token + latent_len
            repr_sum[seq_id, start_token:end_token] += out[j]
            count_sum[seq_id, start_token:end_token] += 1
 
    all_representations = repr_sum / count_sum.clamp(min=1)
    return all_representations