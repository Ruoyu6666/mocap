import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Dataset for loading trained latent representations from a .npy file
class LatentRepresentationDataset(Dataset):
    
    def __init__(self, path_to_latent_representations: str, if_include_test = True):
        self.latent_representations = torch.from_numpy(np.load(path_to_latent_representations)).float()
        #if if_include_test:
        #    path_to_test = path_to_latent_representations.replace("representations_train", "representations_test")
        #    self.latent_representations = torch.cat([self.latent_representations, torch.from_numpy(np.load(path_to_test)).float()], dim=0)
        print(f"Loaded latent representations from {path_to_latent_representations} with shape {self.latent_representations.shape}")

    def __len__(self):
        return len(self.latent_representations)

    def __getitem__(self, idx):
        return self.latent_representations[idx]