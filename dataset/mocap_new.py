import __future__

import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
import torch
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation
from .datasets import BasePoseTrajDataset


class MocapDataset(BasePoseTrajDataset):
    """Primary Mouse (+Features) dataset."""
    DEFAULT_DATASETS = ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]
    
    NUM_INDIVIDUALS = 1
    NUM_KEYPOINTS = 10
    KPTS_DIMENSIONS = 3
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)

    STR_BODY_PARTS = [
        'left_ankle',  'left_back',  'left_coord',  'left_hip',  'left_knee', 
        'right_ankle', 'right_back', 'right_coord', 'right_hip', 'right_knee'
        ]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}


    def __init__(
        self,
        path_to_data_dir: Path,
        datasets: List[str] | None = None, 
        task: str = "CLB",
        sampling_rate: int = 1,
        num_frames: int = 300,
        sliding_window: int = 149,
        interp_holes: bool = False,
        augmentations: transforms.Compose = None,
        left_idx        = 3,       # default left hip
        right_idx       = 8,       # default right hip
        index_frame     = 149, 
        normalizer      = 'normal',
        **kwargs
    ):
        super().__init__(
            path_to_data_dir,
            sampling_rate,
            num_frames,
            sliding_window,
            interp_holes,
            **kwargs
        )
        self.datasets = datasets or self.DEFAULT_DATASETS
        self.task = task
        self.augmentations = augmentations
        self.left_idx      = left_idx
        self.right_idx     = right_idx
        
        self.vi   = ViewInvariant(index_frame = index_frame, left_idx = left_idx, right_idx = right_idx,)
        self.norm = Normalize() if normalizer == 'normal' else NormalizeCube()
        
        self.load_data()
        self.preprocess()

    
    def load_data(self):
        self.raw_data = []
        for dataset_name in self.datasets:
            filepath = Path(self.path_to_data_dir + dataset_name + "_CLB_processed.pkl")
            
            with open(filepath, 'rb') as file:
                self.raw_data.append(pickle.load(file))
        
    def preprocess(self):
        pass
    
    def featurise_keypoints(self, keypoints):
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints