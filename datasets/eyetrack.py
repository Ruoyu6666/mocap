import numpy as np

import torch
from torch.utils.data import Dataset


class EyetrackDataset(Dataset):

    def __init__(self, path_to_data_dir, num_frames):
        data = np.load(path_to_data_dir, allow_pickle=True)
        self.num_frames = num_frames
        X = data["X"]
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx].reshape(self.num_frames, 13, 2)
        seq = torch.unsqueeze(seq, dim = 1)
        return seq, []