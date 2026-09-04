# Sub-sample and get TWO VIEWS
# SHUFFLE within batch
# cluster for initialization

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class ClipDataset(Dataset):
    """
    Loads all clips from a single .npy file, shape (N, T, J, C).
    center_idx defaults to T // 2 for every clip unless a separate (N,) .npy file of per-clip anchor indices provided.
    If your clips live some other way (per-file on disk, generated on-the-fly from long sequences, etc.), replace this class — 
    everything else in the script only depends on __getitem__ returning (clip: (T, J, C) float tensor, center_idx: scalar long tensor).
    """

    def __init__(self, clips_path: str, center_idx_path: str = None):
        self.clips = np.load(clips_path)
        self.clips = torch.from_numpy(self.clips).float()
        N, T, J, C = self.clips.shape
        
        if center_idx_path is not None:
            center_idx = np.load(center_idx_path) #assert center_idx.shape == (N,), (f"center_idx must in shape ({N},), got {center_idx.shape}")
            self.center_idx = torch.from_numpy(center_idx).long()
        else:
            self.center_idx = torch.full((N,), T // 2, dtype=torch.long)

    def __len__(self):
        return self.clips.shape[0]

    def __getitem__(self, idx):
        return self.clips[idx], self.center_idx[idx]

