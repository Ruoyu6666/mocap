import numpy as np
#import plotly.graph_objects as go
import os
import pandas as pd
import torch
import logging



def compute_svd(points):
    """
    points: (n_keypoints=3, 3) numpy array. Should be hip, coord and back to approximate the plane of the mouse back.
    :returns:   barycenter:         (3,) numpy array — mean position of valid points
                transition matrix: (3, 3), both can be in transform_points 
    """
    hip_coord_back = np.full(points.shape, np.nan)
    for i in [1 ,2, 3, 6, 7, 8]:
        hip_coord_back[i,:] = points[i,:]
    points = hip_coord_back[~np.any(np.isnan(hip_coord_back), axis=1)] # the keypoints (of hips, backs, coords) with complete 3d data
    
    if len(points) == 0:
        return np.nan, np.nan
    
    barycenter = np.nanmean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - barycenter)
    
    return barycenter, Vt.T



class ViewInvariant:
    """
    Applies a rotation in the XY plane to make skeleton sequences view-invariant. No norm transformation is applied.

    Strategy:
    - Compute SVD on a reference frame to find the body's principal axes.
    - For standing/walking: use A[:,0] (spine axis) — has large XY component.
    - For climbing:         use A[:,2] (perpendicular to back) — spine is vertical so A[:,0] has near-zero XY component → unstable.
    - Climbing detected when the spine axis (A[:,0]) is dominated by Z.
    - After rotation: body axis aligned with +X (y=0 plane).
    - Facing direction (±X ambiguity) resolved using left/right hip joints.

    Forward pass  (T, J, 3) or (B, T, J, 3):  __call__   → centers + rotates by +angle
    Inverse pass  (B, T, J, 3):                untransform → rotates by -angle + re-adds barycenter
    """
    def __init__(self, index_frame=0, left_idx=None, right_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.index_frame = index_frame
        self.left_idx = left_idx   # e.g., left hip
        self.right_idx = right_idx # e.g., right hip 

    def __str__(self):
        return 'ViewInvariant'

    @staticmethod
    def _rotate_xy(array, angle):
        """
        Apply a 2D rotation of `angle` radians in the XY plane.
        Args:   array: (..., 3), angle: float
        Returns:    rotated array of same shape
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        out = np.empty_like(array)
        x = array[..., 0]
        y = array[..., 1]

        out[..., 0] = x * cos_a - y * sin_a
        out[..., 1] = x * sin_a + y * cos_a
        out[..., 2] = array[..., 2] # Z remains unchanged
        return out

    def _needs_flip(self, rotated_points, A, angle):
        """
        Check if the mouse is facing -X after rotation, needs a 180° flip.
        Uses left/right hip joints: forward = cross(left→right, spine).
        Args:       rotated_points: (J, 3) already rotated + centered reference frame
                    A:              (3, 3) SVD axes of the original frame
                    angle:          float, current rotation angle (before any flip)
        Returns:    bool: True if a 180° flip is needed
        """
        left  = rotated_points[self.left_idx]
        right = rotated_points[self.right_idx]

        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            return False  # can't determine → no flip (safe default)

        lr_vec = right - left
        # Rotate the spine axis by the same angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        spine = A[:, 0].copy()
        spine_rot = np.array([spine[0] * cos_a - spine[1] * sin_a,
                              spine[0] * sin_a + spine[1] * cos_a,
                              spine[2]])
        # Forward direction = cross(left→right, spine)
        forward = np.cross(lr_vec, spine_rot)
        return bool(forward[0] < 0)

    #  Core transform       
    def compute_transform(self, x):
        """
        Compute barycenter and rotation angle from a single reference frame.
        Args:   x: (T, J, 3)
        Returns:    barycenter:  (3,)   centroid used to center all frames
                    index_vect:  int    SVD column used (0 = spine, 2 = dorsal)
                    angle:       float  rotation angle in radians (includes flip if needed)
        """
        # 0. Define essential joints for computing barycenter
        ESSENTIAL_JOINTS = [1, 2, 3, 6, 7, 8]
        def frame_has_essential(frame_points):
            """True if at least one essential joint is fully valid (no NaN)."""
            for j in ESSENTIAL_JOINTS:
                if not np.any(np.isnan(frame_points[j])):  # all 3 coords valid
                    return True
            return False

        # 1. Initial reference frame
        idx    = min(self.index_frame, x.shape[0] - 1)
        points = x[idx]                             # (J, 3) 
        if not frame_has_essential(points):
            valid = [t for t in range(x.shape[0]) if frame_has_essential(x[t])]
            if len(valid) == 0:
                raise ValueError("[ViewInvariant] No frame found where at least one of joints "
                                f"{[j+1 for j in ESSENTIAL_JOINTS]} is valid.")
            
            nan_per_frame = np.sum(np.isnan(x), axis=(1, 2))  # (T,)
            idx     = int(np.argmin(nan_per_frame))     # use frame with fewest NaN joints
            # valid  = np.array(valid)
            points = x[idx]
        
        mask_na = np.any(np.isnan(points), axis=1) # checks per joint whether any of its 3 coordinates is NaN.

        # Final guard: need at least 2 valid joints for SVD to be meaningful
        n_valid = np.sum(~mask_na)
        if n_valid < 2:
            raise ValueError(f"[ViewInvariant] Reference frame {idx} has only {n_valid} valid "
                             f"joint(s) — need at least 2 for SVD.")

        # 2. SVD on clean points
        barycenter, A = compute_svd(points)

        # 3. Detect climbing: spine axis (A[:,0]) dominated by Z → climbing
        max_component_in_A = np.argmax(np.abs(A), axis=0)
        index_vect = 2 if max_component_in_A[0] == 2 else 0 # spine points mostly along Z, use dorsal axis (stable XY when climbing). Otherwise use spine axis (stable XY when walking)

        # 4. Rotation angle to align chosen axis with +X
        vect = A[:, index_vect]
        angle = -np.arctan2(vect[1], vect[0])

        # 5. Check and fix facing direction using left/right hips
        rotated   = self._rotate_xy(points - barycenter, angle)
        if self._needs_flip(rotated, A, angle):
            angle += np.pi  # absorb 180° flip into the angle

        return barycenter, index_vect, angle


    @staticmethod
    def apply_transform(x, barycenter, angle):
        """
        Forward transform: center + rotate by +angle in XY.
        Args:   x:          (T, J, 3) or None
                barycenter: (3,)
                angle:      float
        Returns:    (T, J, 3) transformed, or None
        """
        if x is None:
            return None
        return ViewInvariant._rotate_xy(x - barycenter, angle)

   
    def untransform(self, x, **kwargs):
        """
        Inverse transform: rotate by -angle + re-add barycenter.
        Args:
            x:       (B, T, J, 3) — batched sequences in canonical frame
            kwargs:  must contain 'VI_angle' and 'VI_barycenter'
                     (scalars, numpy arrays, or torch tensors)
        Returns:
            (B, T, J, 3) sequences restored to original coordinate frame
        """
        angle      = kwargs['VI_angle']
        barycenter = kwargs['VI_barycenter']
        """
        if hasattr(angle, 'detach'):            # torch tensor
            angle = angle.detach().cpu().numpy()
        if hasattr(barycenter, 'detach'):
            barycenter = barycenter.detach().cpu().numpy()
        """
        angle      = float(np.squeeze(angle))
        barycenter = np.array(barycenter).reshape(3)  # ensure (3,)

        x_arr = np.array(x)                    # (B, T, J, 3), safe copy
        x_inv = self._rotate_xy(x_arr, -angle) # --- Inverse rotation: apply -angle ---
        x_inv = x_inv + barycenter             # Re-add barycenter

        # Restore positions that were all-zero (missing markers) back to NaN  so downstream code handles them correctly.
        nan_mask = np.all(x_arr == 0, axis=-1)  # (B, T, J) — all coords zero
        x_inv[nan_mask] = np.nan

        return x_inv


    def __call__(self, x, x_supp=(), **kwargs):
        """
        Args:   x:      (T, J, 3) primary sequence
                x_supp: tuple of supplementary sequences, same shape
        Returns:     x_prime:      (T, J, 3) view-invariant primary sequence
                     x_supp_prime: tuple of transformed supplementary sequences
                    kwargs:       updated with VI_barycenter, VI_angle,  min_sample, max_sample
        """
        barycenter, index_vect, angle = self.compute_transform(x)
        x_prime = self.apply_transform(x, barycenter, angle)
        x_supp_prime = tuple(self.apply_transform(xx, barycenter, angle) for xx in x_supp)

        if np.all(np.isnan(x_prime)):
            print('[ViewInvariant] Warning: all NaN in x_prime')

        kwargs['VI_barycenter'] = barycenter
        kwargs['VI_angle']      = angle
        kwargs['min_sample']    = np.nanmin(x_prime, axis=(0, 1))  # (3,)
        kwargs['max_sample']    = np.nanmax(x_prime, axis=(0, 1))  # (3,)

        return x_prime, x_supp_prime, kwargs



class Normalize:
    """
    Per-sample normalization to [-1, 1] independently for each axis (X, Y, Z).
    Min/max are computed from the primary sequence x and applied consistently to all supplementary sequences x_supp.
    """
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return 'Normalize'

    @staticmethod
    def _normalize(x, min_, max_):
        """
        Normalize array to [-1, 1] using provided per-axis min/max. Safe against zero-range axes (returns 0 where max == min).
        Args:    x: (..., 3), min_: (3,), max_: (3,)
        Returns: normalized array, same shape as x
        """
        range_ = max_ - min_
        safe_range = np.where(range_ == 0, 1.0, range_) ## Avoid division by zero: where range is 0, output 0 (midpoint of [-1,1])
        x_norm = 2 * (x - min_) / safe_range - 1
        """
        # Force constant axes to 0 (not ±inf or nan)
        x_norm[..., range_ == 0] = 0.0
        """
        return x_norm

    @staticmethod
    def _unnormalize(x, min_, max_):
        """
        Inverse of _normalize: map [-1, 1] back to original range.
        Args:       x: (..., 3), min_: (3,), max_: (3,)
        Returns:    reconstructed array, same shape as x
        """
        range_ = max_ - min_
        return min_ + range_ * (1 + x) / 2

    def __call__(self, x, x_supp=(), **kwargs):
        """
        Forward normalization.
        Args:       x:      (T, J, 3)
                    x_supp: tuple of supplementary sequences, same shape
        Returns:    x_prime:      (T, J, 3) normalized to [-1, 1]
                    x_supp_prime: tuple of normalized supplementary sequences
        kwargs:     updated with min_sample (3,) and max_sample (3,)
        """
        # Compute per-axis min/max over all time steps and joints
        min_ = np.nanmin(x, axis=(0, 1))              # (3,)
        max_ = np.nanmax(x, axis=(0, 1))              # (3,)

        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print(f'[Normalize] Warning: NaN in min/max — '
                  f'min={min_}, max={max_}. Sequence may be all-NaN.')

        kwargs['min_sample'] = min_
        kwargs['max_sample'] = max_

        # Normalize primary sequence. min_/max_ broadcast naturally over (T, J, 3) → last dim aligns
        x_prime = self._normalize(x, min_, max_)
        # Normalize supplementary sequences with same scale as x
        x_supp_prime = tuple(self._normalize(xx, min_, max_) for xx in x_supp)

        return x_prime, x_supp_prime, kwargs


    def untransform(self, x, **kwargs):
        """
        Inverse normalization: map [-1, 1] back to original coordinate range.
        Args:   x:      (T, J, 3) or (B, T, J, 3)
                kwargs: must contain 'min_sample' (3,) and 'max_sample' (3,)
                    as numpy arrays or torch tensors
        """
        min_sample = kwargs['min_sample']
        max_sample = kwargs['max_sample']
        """
        # Handle torch tensors
        if hasattr(min_sample, 'detach'):
            min_sample = min_sample.detach().cpu().numpy()
        if hasattr(max_sample, 'detach'):
            max_sample = max_sample.detach().cpu().numpy()
        """
        min_sample = np.array(min_sample).reshape(3)  # ensure (3,)
        max_sample = np.array(max_sample).reshape(3)

        if len(x.shape) not in (3, 4):
            raise ValueError(f"[Normalize.untransform] Expected 3D (T,J,3) or 4D (B,T,J,3) "
                             f"input, got shape {x.shape}")

        # min_sample shape (3,) broadcasts over (..., 3) regardless of ndim
        return self._unnormalize(np.array(x), min_sample, max_sample)