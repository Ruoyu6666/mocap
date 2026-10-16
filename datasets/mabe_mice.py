import os
from pathlib import Path
import __future__
import numpy as np

import torch
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation
from .dataset import BasePoseTrajDataset



class MABeMouseDataset(BasePoseTrajDataset):
    """Primary Mouse (+Features) dataset."""
    DEFAULT_GRID_SIZE = 850
    NUM_INDIVIDUALS = 3
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 2
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    DEFAULT_NUM_TRAINING_POINTS = 1600
    DEFAULT_NUM_TESTING_POINTS = 3736
    SAMPLE_LEN = 1800
    NUM_TASKS = 13

    NOSE = "nose"
    EAR_LEFT = "ear_left"
    EAR_RIGHT = "ear_right"
    NECK = "neck"
    FOREPAW_LEFT = "forepaw_left"
    FOREPAW_RIGHT = "forepaw_right"
    CENTER = "center"
    HINDPAW_LEFT = "hindpaw_left"
    HINDPAW_RIGHT = "hindpaw_right"
    TAIL_BASE = "tail_base"
    TAIL_MIDDLE = "tail_middle"
    TAIL_TIP = "tail_tip"
    STR_BODY_PARTS = [NOSE, EAR_LEFT, EAR_RIGHT, 
                      NECK, FOREPAW_LEFT, FOREPAW_RIGHT,
                      CENTER, HINDPAW_LEFT, HINDPAW_RIGHT, 
                      TAIL_BASE, TAIL_MIDDLE, TAIL_TIP,]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    def __init__(self, 
                 mode: str,
                 path_to_data_dir: Path,
                 sampling_rate: int = 1,
                 num_frames: int = 60,
                 sliding_window: int = 10,
                 interp_holes: bool = False,
                 augmentations: bool=False,
                 normalize: bool = True, 
                 centeralign: bool = False,
                 include_testdata: bool = False,
                 model: str = "SkeletonMAE",
                 **kwargs):
        super().__init__(path_to_data_dir, sampling_rate, num_frames, sliding_window, interp_holes, **kwargs)
        
        self.mode = mode
        self.normalize = normalize
        self.centeralign = centeralign
        gs = (self.DEFAULT_GRID_SIZE, self.DEFAULT_GRID_SIZE)
        self.augmentations = augmentations
        self.augment = transforms.Compose([Rotation(grid_size=gs, p=0.5),
                                GaussianNoise(p=0.5), Reflect(grid_size=gs, p=0.5),])
        self.model = model
        self.load_data(include_testdata)
        self.preprocess()


    def load_data(self, include_testdata) -> None:
        if self.mode == "train":
            self.raw_data = np.load(self.path, allow_pickle=True).item()
            if include_testdata:
                raw_data_test = np.load(self.path.replace("train", "test"), allow_pickle=True).item()
                self.raw_data["sequences"].update(raw_data_test["sequences"])
        elif self.mode == "test":
            self.raw_data = np.load("/home/rguo_hpc/myfolder/data/mabe_mice/mouse_triplet_test.npy", allow_pickle=True).item()
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))


    def load_labeled_data(self) -> None:
        self.raw_data = np.load(os.path.join(self.path, "mouse_triplet_test.npy"), allow_pickle=True).item()
        self.frame_number_map = np.load(os.path.join(self.path, "frame_number_map.npy"), allow_pickle=True).item()
        # frame_number_map, label_array, vocabulary, task_type
        self.labels = np.load(os.path.join(self.path, "mouse_triplets_test_labels.npy"), allow_pickle=True).item()


    def preprocess(self):
        """Does initial preprocessing on entire dataset."""
        self.check_annotations()
        sequences = self.raw_data["sequences"]

        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window

        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()):
            vec_seq = sequence["keypoints"] # (1800, 3, 12, 2)
            if self.interp_holes:
                vec_seq = self.fill_holes(vec_seq)
            if self._sampling_rate > 1:
                vec_seq = vec_seq[:: self._sampling_rate]

            if sub_seq_length < 120:
                pad_length = sub_seq_length
            else:
                pad_length = 120
            pad_vec = np.concatenate([np.repeat(vec_seq[:1], pad_length // 2, axis=0), vec_seq, 
                                      np.repeat(vec_seq[-1:], pad_length - pad_length // 2, axis=0)], axis=0)
            seq_keypoints.append(pad_vec)
            keypoints_ids.extend([(seq_ix, i) for i in np.arange(0, len(pad_vec) - sub_seq_length + 1, sliding_window)])

        seq_keypoints = np.array(seq_keypoints, dtype=np.float32)
        self.items = list(np.arange(len(keypoints_ids)))
        self.seq_keypoints = seq_keypoints
        self.keypoints_ids = keypoints_ids
        self.n_frames = len(self.keypoints_ids)
        self.num_sequences = len(seq_keypoints)
        del self.raw_data
        print("Total number of sequences",self.num_sequences, "Total number of subsequences",self.n_frames)


    def prepare_subsequence_sample(self, sequence: np.ndarray):
        if self.augmentations:
            sequence = sequence.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            sequence = self.augment(sequence)
        feats = self.featurise_keypoints(sequence)
        return feats
    
    def mabe_normalize(self,keypoints):
        center =self.DEFAULT_GRID_SIZE / 2
        keypoints = (keypoints - center) / center
        return keypoints
    
    def featurise_keypoints(self, keypoints):
        #keypoints = self.rotate_skeleton(keypoints)
        if self.normalize:
            keypoints = self.mabe_normalize(keypoints)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints

    
    def __getitem__(self, idx: int):
        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len]
        inputs = self.prepare_subsequence_sample(subsequence)
        # [60, 3, 12, 2]
        return inputs, [] # num_samples, sequence_length, num_mice, num_keypoints, 2


    """
    @staticmethod
    def rotate_skeleton(data, neck_idx=3, tail_base_idx=9, center_idx=6):
        #Rotate skeletons to have consistent body orientation. data: (seq_len, num_mice, num_kpts, 2)
        # Reference point: preserve its original position
        center = data[:, :, center_idx, :]
        # Direction from neck to tail base
        neck = data[:, :, neck_idx, :]
        tail = data[:, :, tail_base_idx, :]
        dx = tail[..., 0] - neck[..., 0]
        dy = tail[..., 1] - neck[..., 1]

        # Rotate the neck-to-tail vector to point along +y
        angle = np.pi / 2 - np.arctan2(dy, dx)
        cos_a = np.cos(angle)[..., None]
        sin_a = np.sin(angle)[..., None]

        # Coordinates relative to the center
        offsets = data - center[:, :, None, :]
        x = offsets[..., 0]
        y = offsets[..., 1]
        # Apply the same rotation to every keypoint
        rotated = np.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a], axis=-1)
        # Restore the original center position
        return rotated + center[:, :, None, :]

    @staticmethod
    def mabe_normalize(x):
        # x shape: (seq_len, num_mice, num_kpts, 2)
        min_ = np.nanmin(x, axis=(0, 1, 2))
        max_ = np.nanmax(x, axis=(0, 1, 2))
        range_ = max_ - min_
        invalid_mask = np.isnan(range_) | (range_ < 1e-8)
        safe_range = np.where(invalid_mask, 1.0, range_)
        x_norm = 2 * (x - min_) / safe_range - 1
        # Set constant or invalid coordinate axes to zero
        x_norm[..., invalid_mask] = 0.0
        return x_norm, min_, max_
    """