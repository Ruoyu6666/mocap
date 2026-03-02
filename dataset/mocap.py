import __future__

import os
from pathlib import Path
from typing import List
import numpy as np
import torch
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation
from .pose_traj_dataset import BasePoseTrajDataset



class MocapDataset(BasePoseTrajDataset):
    """Primary Mouse (+Features) dataset."""

    #DEFAULT_FRAME_RATE = 60
    #DEFAULT_GRID_SIZE = 850
    NUM_INDIVIDUALS = 1
    NUM_KEYPOINTS = 10
    KPTS_DIMENSIONS = 3
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    #DEFAULT_NUM_TRAINING_POINTS = 1600
    #DEFAULT_NUM_TESTING_POINTS = 3736
    SAMPLE_LEN = 1800
    #NUM_TASKS = 13

    STR_BODY_PARTS = ['left_ankle',  'left_back',  'left_coord',  'left_hip',  'left_knee', 
                      'right_ankle', 'right_back', 'right_coord', 'right_hip', 'right_knee']
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

        def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        datasets: List[str]: ["CP1A", "CP1B", "INH1", "INH2"], 
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        fill_holes: bool = False,
        augmentations: transforms.Compose = None,
        centeralign: bool = False,
        include_testdata: bool = False,
        **kwargs
    ):

        super().__init__(
            path_to_data_dir,
            scale,
            sampling_rate,
            num_frames,
            sliding_window,
            fill_holes,
            **kwargs
        )

        # self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed
        self.mode = mode
        self.centeralign = centeralign
        if augmentations:
            gs = (self.DEFAULT_GRID_SIZE, self.DEFAULT_GRID_SIZE)
            self.augmentations = transforms.Compose(
                [
                    Rotation(grid_size=gs, p=0.5),
                    GaussianNoise(p=0.5),
                    Reflect(grid_size=gs, p=0.5),
                ]
            )
        else:
            self.augmentations = None

        self.load_data()
        self.preprocess()
    
    
    def load_data(self):
        self.raw_data = []
        for data_name in datasets: 
            with open(os.path.join(self.path, "mouse_triplet_test.npy"), 'rb') as file:
                self.raw_data = pickle.load(file)
        #self.raw_data = np.load(os.path.join(self.path, "mouse_triplet_test.npy"), allow_pickle=True).item()

    def preprocess(self):
        pass
    
    def featurise_keypoints(self, keypoints):
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints
