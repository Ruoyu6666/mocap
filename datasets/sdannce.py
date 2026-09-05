import __future__

import pickle
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

from .transform import NormalizeConfig,  rotation_normalize, ViewInvariant
from .augmentations import Augmentations, _resample_time
from .dataset import BasePoseTrajDataset


# Noramlize Config
NormalizeConfig_sdannce = NormalizeConfig(left_shoulder=6, right_shoulder=9, left_hip=12, right_hip =15)


class SdannceDataset(BasePoseTrajDataset):
    NUM_KEYPOINTS = 18
    STR_BODY_PARTS = ["Snout", "EarL", "EarR",  
                      "SpineF", "SpineM",  "TailBase",
                      "ShoulderL", "ElbowL", "WristL", 
                      "ShoulderR", "ElbowR", "WristR", 
                      "HipL", "KneeL", "AnkleL", 
                      "HipR", "KneeR", "AnkleR"] 
    NUM_INDIVIDUALS = 1
    SAMPLE_LEN = 4500
    KPTS_DIMENSIONS = 3
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}
    Normalize_Config = NormalizeConfig_sdannce
    
    def __init__(self, 
                mode: str, 
                path_to_data_dir: Path,
                sampling_rate: int = 1,
                num_frames: int = 300,
                sliding_window: int = 100,
                interp_holes: bool = False,
                view_invariant: bool = True,
                augmentations: bool = False, 
                normalize: bool = True, 
                model: str = "SkeletonMAE",
                split: dict = None,     # Split dataset by mouse for train/val. No split if None
                if_val: bool = False,   # When split not None: if False, load mouse for training, if true, load mouse for validation
                **kwargs):
        
        super().__init__(path_to_data_dir, sampling_rate, num_frames, sliding_window, interp_holes, **kwargs)

        self.mode = mode
        self.view_invariant = view_invariant
        if view_invariant:
            self.vi = ViewInvariant(index_frame = num_frames//2, left_idx = self.Normalize_Config.left_hip, right_idx = self.Normalize_Config.right_hip)
        if augmentations:
            self.augmentations = Augmentations()
        self.normalize = normalize

        self.model = model
        self.split = split
        self.if_val = if_val
        self.load_data()
        self.preprocess()


    def load_data(self):
        with open(self.path, 'rb') as file:
            result = pickle.load(file)
        L = self.SAMPLE_LEN
        N = int(90000 /self.SAMPLE_LEN) # number of sequences per huge sequence
        
        for mouse in result.keys(): # for each mouse
            num_seq = len(result[mouse]["ratgen"])
            ratgen  = int(result[mouse]["ratgen"][0])
            result[mouse]["ratgen"] = np.full((num_seq * N,), ratgen)

            result[mouse]["m1"] = np.array(result[mouse]["m1"])     # (3, 90000, 3, 23)
            result[mouse]["m1"] = result[mouse]["m1"].transpose(0, 1, 3, 2).reshape(-1, L, 23, 3)
            result[mouse]["m1"] = np.delete(result[mouse]["m1"], [5, 10, 14, 18, 22], axis=-2)
            
            result[mouse]["llac"] = np.array(result[mouse]["llac"]) # (3, 90000, 1)
            result[mouse]["llac"] = np.squeeze(result[mouse]["llac"], axis=2)
            result[mouse]["llac"] = result[mouse]["llac"].reshape(-1, L,)
            
            result[mouse]["hlac"] = np.array(result[mouse]["hlac"]) # (3, 90000, 1)
            result[mouse]["hlac"] = np.squeeze(result[mouse]["hlac"], axis=2)
            result[mouse]["hlac"] = result[mouse]["hlac"].reshape(-1, L,)
        
        self.raw_data = result
        del result


    def preprocess(self):
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        half_len = sub_seq_length// 2

        num_sequences_total = 0
        if self.split is not None: # only take train/test data
            if self.if_val:
                mice = self.split["valid"]
            else:
                mice = self.split["train"]
        else: 
            mice = self.raw_data.keys() # if self.split is None, use data of all mice
        for mouse_name in mice: # Iterate over mice
            sequences = self.raw_data[mouse_name]["m1"] #(num_sequences, 3600, 10, 3)
            num_sequences = len(sequences)
            if True:
            #if self.mode in ["pretrain"]: # padding
                for i in range(num_sequences_total, num_sequences_total + num_sequences): # Iterate over sequences of one mouse
                    vec_seq = sequences[i-num_sequences_total]
                    pad_vec = np.pad(vec_seq, ((half_len, sub_seq_length - half_len), (0, 0), (0, 0)), mode="edge", ) # Pads the beginning and end of the sequence with duplicate frames
                    seq_keypoints.append(pad_vec)
                    keypoints_ids.extend([(i, sub_i) for sub_i in np.arange(0, len(pad_vec) - sub_seq_length + 1, self.sliding_window)])
            """
            elif self.mode in ["compute_representations", "linprobe", "finetune"]: # no paddiing
                for i in range(num_sequences_total, num_sequences_total + num_sequences):
                    vec_seq = sequences[i-num_sequences_total]
                    seq_keypoints.append(vec_seq)
                    keypoints_ids.extend([(i, sub_i) for sub_i in np.arange(0, len(vec_seq), self.sliding_window)])
            """
            num_sequences_total += num_sequences
        
        self.num_sequences = num_sequences_total
        self.seq_keypoints = np.array(seq_keypoints, dtype=np.float32) # numpy array: [num_sequences, T/5 + pad_vect, 10, 3]
        self.keypoints_ids = keypoints_ids
        print("Total number of sequences",self.num_sequences, "Total number of subsequences",len(self.keypoints_ids))

        del self.raw_data

    def featurise_keypoints(self, keypoints):
        seq = keypoints.reshape(-1, self.NUM_KEYPOINTS, 3)
        if self.interp_holes:
            seq = self.fill_holes(seq)
        if self.view_invariant:         # View-invariant transformation
            seq, _, _  = self.vi(seq, x_supp=(),)
        if self.augmentations:          # Augmentations
            seq = self.augmentations(seq)
            seq = seq[::5] 
            seq =  _resample_time(seq, target_len=self.max_keypoints_len)
        if self.normalize:              # Normalize
            seq, _, _ = self.mocap_normalize(seq)
        seq = torch.tensor(seq, dtype=torch.float32)
        seq = torch.nan_to_num(seq, nan = 0.0) # replace NaN with 0.0
        return seq
    
    
    @staticmethod
    def mocap_normalize(x):
        """
        Per-sample normalization to [-1, 1] independently per axis (X, Y, Z). Does NOT preserve aspect ratio — each axis uses its own min/max.
        Args:       x: (T, J, 3)
        Returns:    x_norm: (T, J, 3) normalized to [-1, 1]; min_, max_: (3,) per-axis minimum (needed for untransform)
        """
        min_ = np.nanmin(x, axis=(0, 1))              # (3,)
        max_ = np.nanmax(x, axis=(0, 1))              # (3,)
        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print('[Normalize] Warning: NaN in min/max — sequence may be all-NaN.')
        range_     = max_ - min_                      # (3,)
        invalid_mask = np.isnan(range_) | (range_ < 1e-8)  # Identify invalid/flat axes (zero or NaN range)
        safe_range = np.where(invalid_mask, 1.0, range_) # Use np.where to prevent 0/0 or nan/nan division warnings    
        x_norm = 2 * (x - min_) / safe_range - 1
        x_norm[..., invalid_mask] = 0.0 # Zero out axes that were constant or all-NaN

        return x_norm, min_, max_
    

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
                if first_valid > 0:                                  # fill with first valid value
                    filled[:first_valid, j, c] = values[first_valid]
                for i in range(first_valid, self.max_keypoints_len): # fill with last valid value
                    if not valid[i]:
                        filled[i, j, c] = filled[i-1, j, c]
        
        return filled
    

    def prepare_subsequence_sample(self, sequence: np.ndarray):  # sequence (300, 23, 3)
        """Returns one training sample"""
        feats = self.featurise_keypoints(sequence) # [300, 1, 23, 3]
        if self.model == "behaveMAE":
            feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1) # [300, num_ind, num_joints* 2d] flatten for behaveMAE

        return feats
    

    def __getitem__(self, idx: int):
        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len]
        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []