import __future__

import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
import torch
from torchvision import transforms
from .transform import ViewInvariant
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
        mode: str, 
        path_to_data_dir: Path,
        datasets: List[str] | None = None, 
        task: str = "CLB" , # FL2 or Tr
        sampling_rate: int = 1,
        num_frames: int = 300,
        sliding_window: int = 149,
        interp_holes: bool = False,
        augmentations: transforms.Compose = None,
        view_invariant: bool = True, 
        left_idx:     int = 3,       # default left hip
        right_idx:    int = 8,       # default right hip
        index_frame:  int = 149, 
        normalizer:    str = 'normal',
        model: str = "SkeletonMAE",
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
        self.mode = mode
        self.datasets = datasets or self.DEFAULT_DATASETS
        self.task = task
        self.augmentations = augmentations
        self.left_idx      = left_idx
        self.right_idx     = right_idx
        
        if view_invariant:
            self.vi  = ViewInvariant(index_frame = index_frame, left_idx = left_idx, right_idx = right_idx,)
        # self.norm = Normalize() if normalizer == 'normal' else NormalizeCube()
        
        if augmentations:
            self.augmentations = transforms.Compose([GaussianNoise(p=0.5),])
        
        self.model = model
        
        self.load_data()
        self.preprocess()


    # Step 0: load raw data
    def load_data(self):
        with open(self.path, 'rb') as file:
                self.raw_data = pickle.load(file)

    def preprocess(self):
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        

        num_sequences_total = 0
        for dataset_name in self.datasets:
            sequences = self.raw_data[dataset_name]["data"] #(num_sequences, 3600, 10, 3)
            num_sequences = len(sequences)
            if self.mode == "pretrain":
            # For each dataset
                for i in range(num_sequences_total, num_sequences_total + num_sequences):
                    vec_seq = sequences[i-num_sequences_total]
                    # Pads the beginning and end of the sequence with duplicate frames
                    pad_vec = np.pad(vec_seq, ((sub_seq_length// 2, sub_seq_length - sub_seq_length // 2), (0, 0), (0, 0)), mode="edge", )
                    seq_keypoints.append(pad_vec)
                    # For extracting subsequenes
                    keypoints_ids.extend([(i, sub_i) for sub_i in np.arange(0, len(pad_vec) - sub_seq_length + 1, self.sliding_window)])
            elif self.mode == "compute_representations":
                for i in range(num_sequences_total, num_sequences_total + num_sequences):
                    vec_seq = sequences[i-num_sequences_total]
                    seq_keypoints.append(vec_seq)
                    keypoints_ids.extend([(i, sub_i) for sub_i in np.arange(0, len(vec_seq), self.sliding_window)])

            num_sequences_total += num_sequences
        
        self.seq_keypoints = np.array(seq_keypoints, dtype=np.float32) # numpy array: [num_sequences, T/5 + pad_vect, 10, 3]
        self.keypoints_ids = keypoints_ids

        del self.raw_data

        
    def featurise_keypoints(self, keypoints):
        # Step 2: Apply ViewInvariant → Normalize to a single subsequence.
        if self.model == "SkeletonMAE":
            keypoints = keypoints.reshape(-1, 10, 3)
            
        seq, _, _  = self.vi(keypoints,   x_supp=(),)
        seq, _, _ = self.mocap_normalize(seq)
        seq = torch.tensor(seq, dtype=torch.float32)
        
        if self.model == "SkeletonMAE":
            seq  = torch.unsqueeze(seq, dim = 1)
            seq = torch.nan_to_num(seq)
        
        return seq
    
    
    @staticmethod
    def mocap_normalize(x):
        """
        Per-sample normalization to [-1, 1] independently per axis (X, Y, Z). Does NOT preserve aspect ratio — each axis uses its own min/max.

        Args:       x: (T, J, 3)
        Returns:    x_norm:  (T, J, 3) normalized to [-1, 1]
                    min_, max_:    (3,) per-axis minimum (needed for untransform)
        """
        min_ = np.nanmin(x, axis=(0, 1))              # (3,)
        max_ = np.nanmax(x, axis=(0, 1))              # (3,)

        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print('[Normalize] Warning: NaN in min/max — sequence may be all-NaN.')

        range_     = max_ - min_                      # (3,)
        safe_range = np.where(range_ == 0, 1.0, range_)
        x_norm     = 2 * (x - min_) / safe_range - 1
        x_norm[..., range_ == 0] = 0.0               # constant axis → 0 (midpoint)

        return x_norm, min_, max_
    

    @staticmethod
    def mocap_unnormalize(x, min_, max_):
        """
        Inverse of normalize: restore original coordinate range.
        Returns: (T, J, 3) or (B, T, J, 3) restored array
        """
        min_ = np.array(min_).reshape(3)
        max_ = np.array(max_).reshape(3)
        return min_ + (max_ - min_) * (1 + x) / 2
    

    @staticmethod
    def normalize_cube(x):
        """
        Per-sample normalization to [-1, 1] using ONE scale factor across all axes.
        Preserves aspect ratio — fits skeleton in a cube.

        Args:       x: (T, J, 3)
        Returns:    x_norm:     (T, J, 3) normalized
                    min_, max_: (3,) per-axis minimum (needed for untransform)
        """
        min_ = np.nanmin(x, axis=(0, 1))              # (3,)
        max_ = np.nanmax(x, axis=(0, 1))              # (3,)

        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print('[NormalizeCube] Warning: NaN in min/max — sequence may be all-NaN.')

        amplitude = np.max(max_ - min_)               # scalar — largest range wins
        center    = (min_ + max_) / 2                 # (3,)

        if amplitude == 0:
            return np.zeros_like(x), min_, max_
        x_norm = 2 * (x - center) / amplitude

        return x_norm, min_, max_



    @staticmethod
    def unnormalize_cube(x, min_, max_):
        """
        Inverse of normalize_cube: restore original coordinate range.
        """
        min_ = np.array(min_)
        max_ = np.array(max_)

        if min_.ndim == 2:                            # (B, 3) → (B, 1, 1, 3)
            min_ = min_[:, None, None, :]
            max_ = max_[:, None, None, :]

        amplitude = np.max(max_ - min_, axis=-1, keepdims=True)
        center    = (min_ + max_) / 2

        return amplitude / 2 * x + center
    

    def prepare_subsequence_sample(self, sequence: np.ndarray):
        """Returns one training sample"""
        if self.augmentations:
            sequence = sequence.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            sequence = self.augmentations(sequence)
            sequence = sequence.reshape(self.max_keypoints_len, -1)
        feats = self.featurise_keypoints(sequence) # [300, 1, 10, 3]
        # flatten for now
        #feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1) # [300, num_ind, num_joints* 2d or 3d]

        return feats