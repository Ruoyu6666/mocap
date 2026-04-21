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
    SAMPLE_LEN = 3600

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
        #task: str = "CLB" , # FL2 or Tr
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
        split: dict = None, # whether to split dataset by mouse for train/val
        if_val: bool = False,   # When split is not None: if False, load mouse for training, if true, load mouse for validation mice
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
        #self.task = task
        self.augmentations = augmentations
        
        self.view_invariant = view_invariant
        self.vi  = ViewInvariant(index_frame = index_frame, left_idx = left_idx, right_idx = right_idx,)
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.normalizer = normalizer
        if augmentations:
            self.augmentations = transforms.Compose([GaussianNoise(p=0.5),])
        
        self.model = model
        self.split = split
        self.if_val = if_val
        
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
        for dataset_name in self.datasets: # iterate through datasets ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]
            if self.split is not None:
                if self.if_val:
                    mice = self.split[dataset_name]["valid"]
                else:
                    mice = self.split[dataset_name]["train"]
            else: # use all data if self.split is None
                mice = self.raw_data[dataset_name].keys()
            
            for mouse_name in mice:
                sequences = self.raw_data[dataset_name][mouse_name]["data"] #(num_sequences, 3600, 10, 3)
                num_sequences = len(sequences)
                #if self.mode == "pretrain":
                if True:
                    for i in range(num_sequences_total, num_sequences_total + num_sequences):
                        vec_seq = sequences[i-num_sequences_total]
                        # Pads the beginning and end of the sequence with duplicate frames
                        pad_vec = np.pad(vec_seq, ((sub_seq_length// 2, sub_seq_length - sub_seq_length // 2), (0, 0), (0, 0)), mode="edge", )
                        """
                        # normalize and make view-invariant on the whole sequence before extracting subsequences
                        if self.view_invariant:
                            pad_vec, _, _  = self.vi(pad_vec, x_supp=(),)
                        if self.normalizer == 'normal':
                            pad_vec, _, _ = self.mocap_normalize(pad_vec)
                        elif self.normalizer == 'cube':
                            pad_vec, _, _ = self.normalize_cube(pad_vec)
                        """
                        seq_keypoints.append(pad_vec)
                        # For extracting subsequenes
                        keypoints_ids.extend([(i, sub_i) for sub_i in np.arange(0, len(pad_vec) - sub_seq_length + 1, self.sliding_window)])
                """
                else: #self.mode in ["compute_representations","linprobe", "finetune"]:
                    for i in range(num_sequences_total, num_sequences_total + num_sequences):
                        vec_seq = sequences[i-num_sequences_total]
                        seq_keypoints.append(vec_seq)
                        keypoints_ids.extend([(i, sub_i) for sub_i in np.arange(0, len(vec_seq), self.sliding_window)])
                """
                num_sequences_total += num_sequences
        
        self.num_sequences = num_sequences_total
        self.seq_keypoints = np.array(seq_keypoints, dtype=np.float32) # numpy array: [num_sequences, T/5 + pad_vect, 10, 3]
        self.keypoints_ids = keypoints_ids
        print(self.num_sequences)
        print(len(self.keypoints_ids))

        del self.raw_data

        
    def featurise_keypoints(self, keypoints):
        # Step 2: Apply ViewInvariant → Normalize to a single subsequence.
        #if self.model == "SkeletonMAE":
        seq = keypoints.reshape(-1, 10, 3)
        if self.interp_holes:
            seq = self.fill_holes(seq)
        
        if self.view_invariant:
            seq, _, _  = self.vi(seq, x_supp=(),)
        if self.normalizer == 'normal':
            seq, _, _ = self.mocap_normalize(seq)
        elif self.normalizer == 'cube':
            seq, _, _ = self.normalize_cube(seq)
        
        seq = torch.tensor(seq, dtype=torch.float32)
        
        if self.model == "SkeletonMAE":
            seq  = torch.unsqueeze(seq, dim = 1)
            seq = torch.nan_to_num(seq) # replace NaN with 0.0
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
        """Inverse of normalize_cube: restore original coordinate range."""
        min_ = np.array(min_)
        max_ = np.array(max_)
        if min_.ndim == 2:                            # (B, 3) → (B, 1, 1, 3)
            min_ = min_[:, None, None, :]
            max_ = max_[:, None, None, :]
            amplitude = np.max(max_ - min_, axis=-1, keepdims=True)  # (B, 1, 1, 1)
        else:                                       # (3,) → (1, 1, 3)
            amplitude = np.max(max_ - min_, axis=-1, keepdims=True)
        center   = (min_ + max_) / 2

        return amplitude / 2 * x + center
    

    def fill_holes(self, sequence):
        """Interpolate NaN holes in a single sequence (T, J, 3) using linear interpolation along the time axis."""
        filled = sequence.copy()
        for j in range(self.NUM_KEYPOINTS):
            for c in range(self.KPTS_DIMENSIONS):
                values = filled[:, j, c]
                valid = ~np.isnan(values)
                if np.sum(valid) == 0:  # completely missing keypoint or no missing → skip (leave as NaN, to be handled by model or loss)
                    continue
                first_valid = np.where(valid)[0][0]
                if first_valid > 0:  # leading holes → fill with first valid value
                    filled[:first_valid, j, c] = values[first_valid]
                for i in range(1, self.max_keypoints_len):
                    if not valid[i]:  # hole → fill with last valid value
                        filled[i, j, c] = filled[i-1, j, c]
                """
                filled[:, j, c] = np.interp(
                    np.arange(self.max_keypoints_len),
                    np.where(valid)[0],
                    values[valid]
                )
                """
        return filled
    


    def prepare_subsequence_sample(self, sequence: np.ndarray):  # sequence (300, 10, 3)
        """Returns one training sample"""
        if self.augmentations:
            sequence = sequence.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            sequence = self.augmentations(sequence)
            sequence = sequence.reshape(self.max_keypoints_len, -1)
        feats = self.featurise_keypoints(sequence) # [300, 1, 10, 3]
        if self.model == "behaveMAE":
            feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1) # [300, num_ind, num_joints* 2d] flatten for behaveMAE

        return feats
    

    def __getitem__(self, idx: int):
        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len]
        inputs = self.prepare_subsequence_sample(subsequence)
        
        return inputs, []