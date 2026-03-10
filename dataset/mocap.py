import __future__

import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
import torch
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation



class MocapDataset(torch.utils.data.Dataset):

    NUM_INDIVIDUALS = 1
    NUM_KEYPOINTS = 10
    KPTS_DIMENSIONS = 3
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)

    STR_BODY_PARTS = [
        'left_ankle',  'left_back',  'left_coord',  'left_hip',  'left_knee', 
        'right_ankle', 'right_back', 'right_coord', 'right_hip', 'right_knee'
        ]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    def __init__(self,
                path_to_data_dir: Path,
                datasets: List[str] | None = None, # ["CP1A", "CP1B", "INH1", "INH2"],
                sampling_rate: int = 1,
                num_frames: int = 300,
                sliding_window: int = 149,
                fill_holes: bool = False,
                augmentations: transforms.Compose = None,
                **kwargs
    ):
        self.path_to_data_dir = path_to_data_dir
        self.datasets = ["CP1A","CP1B","INH1","INH2"]

        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        self.sliding_window = sliding_window
        self.fill_holes = fill_holes

        self.load_data()
        #self.preprocess()
    
    
    def load_data(self):
        with open(self.path_to_data_dir, 'rb') as file:
            self.data = pickle.load(file)
    """
        self.raw_data = []
        for  dataset_name in self.datasets:
            filepath = Path(self.path_to_data_dir + dataset_name + "_CLB_processed.pkl")
            with open(filepath, 'rb') as file:
                self.raw_data.append(pickle.load(file))
        
    def preprocess(self):
        pass
    
    def featurise_keypoints(self, keypoints):
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints
    """
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx: int):
        input = self.data[idx]
        input = input.reshape(-1, 30)
        #print(sum(np.isnan(input)))
        input =  torch.tensor(input, dtype=torch.float32)
        input = torch.unsqueeze(input, axis=1)
        
        return input, []
