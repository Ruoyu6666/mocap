"""
Augmentation-based view generation for skeleton SSL (SwAV-style).

Operates on ONE clip at a time: clip shape (T, V, 3), T variable per clip. 
Each call to `generate_view` / `generate_multicrop_views` returns fixed-length view(s), 
resampled via linear interpolation along time -- so it's agnostic to the input clip's original length. 
Plug this into your Dataset.__getitem__, upstream of batching.

Assumes input clips are already normalized (centered + yaw-aligned + scaled). Here are *additional* random perturbations 
layered on top of that canonical pose, to create distinctpositive views for contrastive/clustering (SwAV) training.
"""
import __future__
import numpy as np
from typing import List, Tuple



class Augmentations:
    def __init__(self, jitter_std: float = 2, jitter_p: float = 0.5,
                rotation_range: float = np.pi, rotation_p: float = 0.5,
                reflect_p: float = 0.5,
                scale_range: Tuple[float, float] = (0.9, 1.1), scale_p: float = 0.5,):
        
        self.jitter_std = jitter_std
        self.jitter_p = jitter_p
        self.rotation_range = rotation_range
        self.rotation_p = rotation_p
        self.reflect_p = reflect_p
        #self.scale_range = scale_range
        #self.scale_p = scale_p
        

    # 1. add gaussian noise
    def random_jitter(self, clip: np.ndarray, std: float, p: float) -> np.ndarray:
        if np.random.random() > p:
            return clip
        noise = np.random.normal(loc=0.0, scale=std, size=clip.shape).astype(clip.dtype)
        return clip + noise

    # 2. rotate
    def random_rotate(self, keypoints: np.ndarray, p: float = 0.5, rotation_range: float = np.pi) -> np.ndarray:
        """Randomly rotate a (T, J, 2/3) trajectory around its own centroid."""
        if np.random.random() > p:
            return keypoints
        rot_kpts = keypoints.copy()
        xy = rot_kpts[..., :2]
        center = np.array([np.mean(xy[..., 0]), np.mean(xy[..., 1])])

        angle = np.random.uniform(low=-rotation_range, high=rotation_range)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        rotated_xy = (xy - center) @ R.T + center
        rot_kpts[..., :2] = rotated_xy
        return rot_kpts
    

    # 3. random reflect
    def _reflect_points(self, points: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        """Reflect x,y points across the line A*x + B*y + C = 0."""
        M = np.sqrt(A * A + B * B)
        A, B, C = A / M, B / M, C / M
        D = A * points[..., 0] + B * points[..., 1] + C

        new_points = points.copy()
        new_points[..., 0] = points[..., 0] - 2 * A * D
        new_points[..., 1] = points[..., 1] - 2 * B * D
        return new_points


    def random_reflect(self, keypoints: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly reflect across a horizontal or vertical line through centroid."""
        if np.random.random() > p:
            return keypoints
        new_keypoints = keypoints.copy()
        xy = new_keypoints[..., :2]
        center_x = np.mean(xy[..., 0])
        center_y = np.mean(xy[..., 1])
        if np.random.random() > 0.5:
            reflected_xy = self._reflect_points(xy, 0, 1, -center_y)
        else:
            reflected_xy = self._reflect_points(xy, 1, 0, -center_x)
        new_keypoints[..., :2] = reflected_xy
        return new_keypoints


    def __call__(self, clip: np.ndarray) -> np.ndarray:
            clip = self.random_jitter(clip, self.jitter_std, self.jitter_p)
            clip = self.random_rotate(clip, self.rotation_p, self.rotation_range)
            clip = self.random_reflect(clip, self.reflect_p)
            #clip = self.random_scale(clip, self.scale_range, self.scale_p)
    
            return clip
    """
    # 4. random scale
    def random_scale(self, clip: np.ndarray, scale_range: Tuple[float, float], p: float = 0.5) -> np.ndarray:
        #Randomly scale a clip by a factor drawn from scale_range
        if np.random.random() > p:
            return clip
        s = np.random.uniform(*scale_range)
        return clip * s
    """




#########################
##### Temporal Aug.######
#########################
def _resample_time(clip: np.ndarray, target_len: int) -> np.ndarray:
    """
    Linearly interpolate a clip along the time axis to target_len frames. 
    Works for arbitrary T (including T < target_len upsampling)
    """
    T, V, C = clip.shape
    if T == target_len:
        return clip
    src_idx = np.linspace(0, T - 1, num=T)
    tgt_idx = np.linspace(0, T - 1, num=target_len)

    flat = clip.reshape(T, V * C)
    out = np.empty((target_len, V * C), dtype=clip.dtype)
    for j in range(V * C):
        out[:, j] = np.interp(tgt_idx, src_idx, flat[:, j])
    return out.reshape(target_len, V, C)


def _random_temporal_crop(clip: np.ndarray, crop_ratio_range: Tuple[float, float]) -> np.ndarray:
    T = clip.shape[0]
    ratio = np.random.uniform(*crop_ratio_range)
    crop_len = max(2, int(round(T * ratio)))
    crop_len = min(crop_len, T)
    max_start = T - crop_len
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    return clip[start:start + crop_len]



def generate_view(clip: np.ndarray, target_len: int, aug: Augmentations = Augmentations()) -> np.ndarray:
    """
    Produce ONE augmented view of `clip` (T, V, 3), resampled to `target_len` frames.
    temporal crop -> resample to target_len -> spatial augmentations.
    """
    v = _random_temporal_crop(clip, aug.crop_ratio_range)
    """
    # optional speed jitter: resample crop to a intermediate length before resize, so the *effective playback speed* varies too.
    jitter = np.random.uniform(*cfg.speed_jitter_range)
    interim_len = max(2, int(round(target_len * jitter)))
    v = _resample_time(v, interim_len)
    """
    v = _resample_time(v, target_len)
    v = aug(v)
    return v




def generate_multicrop_views(clip: np.ndarray, global_len: int=64, local_len: int=32, 
                             num_global: int=2, num_local: int=4,
                             global_crop_ratio: Tuple[float, float] = (0.85, 1.0), local_crop_ratio: Tuple[float, float] = (0.3, 0.6),
                             cfg: Augmentations = Augmentations()) -> List[np.ndarray]:
    """
    SwAV-style multi-crop: `num_global` mildly-augmented near-full-length views + `num_local` more aggressively-cropped short views. 
                            Returns a list of arrays, each (T_i, V, 3) -- global views have T=global_len, local views T=local_len.

    Use the global views for the "teacher" side of the swapped assignment (Q side, i.e. sinkhorn targets) 
    and all views (global + local) for the student/prediction side, per the original SwAV recipe.
    """
    import copy
    global_cfg = copy.deepcopy(cfg)
    global_cfg.crop_ratio_range = global_crop_ratio

    local_cfg = copy.deepcopy(cfg)
    local_cfg.crop_ratio_range = local_crop_ratio
    local_cfg.jitter_std = cfg.jitter_std * 1.5      # slightly stronger noise on local crops
    local_cfg.joint_dropout_prob = min(1.0, cfg.joint_dropout_prob * 1.5)

    views = [generate_view(clip, global_len, global_cfg) for _ in range(num_global)]
    views += [generate_view(clip, local_len, local_cfg) for _ in range(num_local)]
    return views