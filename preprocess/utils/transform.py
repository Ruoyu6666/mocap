import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd
import torch
import logging

from utils import compute_svd




class ViewInvariant(Transform):
    """
    Applies a rotation in the XY plane to make skeleton sequences view-invariant. No norm transformation is applied.

    Strategy:
    - Compute SVD on a reference frame to find the body's principal axes.
    - For standing/walking: use A[:,0] (spine axis) — has large XY component.
    - For climbing:         use A[:,2] (perpendicular to back) — spine is vertical
                            so A[:,0] has near-zero XY component → unstable.
    - Climbing detected when the spine axis (A[:,0]) is dominated by Z.
    - After rotation: body axis aligned with +X (y=0 plane).
    - Facing direction (±X ambiguity) resolved using left/right hip joints.

    Forward pass  (T, J, 3) or (B, T, J, 3):  __call__   → centers + rotates by +angle
    Inverse pass  (B, T, J, 3):                untransform → rotates by -angle + re-adds barycenter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index_frame = kwargs['index_frame']
        self.left_idx    = kwargs['left_idx']   # e.g. left hip joint index
        self.right_idx   = kwargs['right_idx']  # e.g. right hip joint index

    def __str__(self):
        return 'ViewInvariant'

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotate_xy(array, angle):
        """
        Apply a 2D rotation of `angle` radians in the XY plane.
        Works on any array whose last dimension is >= 2 (X, Y, [Z, ...]). Z and higher dimensions are left unchanged.
        Args:
            array: (..., 3)
            angle: float
        Returns:
            rotated array of same shape
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        out = array.copy()
        out[..., 0] = array[..., 0] * cos_a - array[..., 1] * sin_a
        out[..., 1] = array[..., 0] * sin_a + array[..., 1] * cos_a
        return out

    def _needs_flip(self, rotated_points, A, angle):
        """
        Check if the mouse is facing -X after rotation, needs a 180° flip.
        Uses left/right hip joints: forward = cross(left→right, spine).

        Args:
            rotated_points: (J, 3) already rotated + centered reference frame
            A:              (3, 3) SVD axes of the original frame
            angle:          float, current rotation angle (before any flip)
        Returns:
            bool: True if a 180° flip is needed
        """
        left  = rotated_points[self.left_idx]
        right = rotated_points[self.right_idx]

        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            return False  # can't determine → no flip (safe default)

        lr_vec = right - left
        # Rotate the spine axis by the same angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        spine = A[:, 0].copy()
        spine_rot = np.array([
            spine[0] * cos_a - spine[1] * sin_a,
            spine[0] * sin_a + spine[1] * cos_a,
            spine[2]
        ])
        # Forward direction = cross(left→right, spine)
        forward = np.cross(lr_vec, spine_rot)
        return bool(forward[0] < 0)

    # ------------------------------------------------------------------ #
    #  Core transform                                                      #
    # ------------------------------------------------------------------ #

    def compute_transform(self, x):
        """
        Compute barycenter and rotation angle from a single reference frame.
        Args:
            x: (T, J, 3)
        Returns:
            barycenter:  (3,)   centroid used to center all frames
            index_vect:  int    SVD column used (0 = spine, 2 = dorsal)
            angle:       float  rotation angle in radians (includes flip if needed)
        """
        # 1. Pick reference frame, fallback to nearest valid frame if needed
        idx = min(self.index_frame, x.shape[0] - 1)
        points = x[idx]
        mask_na = np.any(np.isnan(points), axis=1)

        if np.all(mask_na):
            valid = np.where(np.sum(np.isnan(x), axis=(1, 2)) == 0)[0]
            if len(valid) == 0:
                raise ValueError("No fully valid frame found in sequence.")
            idx = valid[np.argmin(np.abs(valid - self.index_frame))]
            points = x[idx]
            mask_na = np.any(np.isnan(points), axis=1)

        points_clean = points[~mask_na]

        # 2. SVD on clean points
        barycenter, A = compute_svd(points_clean)

        # 3. Detect climbing: spine axis (A[:,0]) dominated by Z → climbing
        max_component_in_A = np.argmax(np.abs(A), axis=0)
        if max_component_in_A[0] == 2:  # spine points mostly along Z
            index_vect = 2              # use dorsal axis (stable XY when climbing)
        else:
            index_vect = 0              # use spine axis (stable XY when walking)

        # 4. Rotation angle to align chosen axis with +X
        vect = A[:, index_vect]
        angle = -np.arctan2(vect[1], vect[0])

        # 5. Check and fix facing direction using left/right hips
        centered  = points - barycenter
        rotated   = self._rotate_xy(centered, angle)
        if self._needs_flip(rotated, A, angle):
            angle += np.pi  # absorb 180° flip into the angle

        return barycenter, index_vect, angle

    @staticmethod
    def apply_transform(x, barycenter, angle):
        """
        Forward transform: center + rotate by +angle in XY.
        Args:
            x:          (T, J, 3) or None
            barycenter: (3,)
            angle:      float
        Returns:
            (T, J, 3) transformed, or None
        """
        if x is None:
            return None
        centered = x - barycenter
        return ViewInvariant._rotate_xy(centered, angle)

    # ------------------------------------------------------------------ #
    #  Inverse transform                                                   #
    # ------------------------------------------------------------------ #

    def untransform(self, x, *args, **kwargs):
        """
        Inverse transform: rotate by -angle + re-add barycenter.
        Args:
            x:       (B, T, J, 3) — batched sequences in canonical frame
            kwargs:  must contain 'VI_angle' and 'VI_barycenter'
                     (scalars, numpy arrays, or torch tensors)
        Returns:
            (B, T, J, 3) sequences restored to original coordinate frame
        """
        # --- Unpack, handle both numpy and torch tensors ---
        angle      = kwargs['VI_angle']
        barycenter = kwargs['VI_barycenter']

        if hasattr(angle, 'detach'):            # torch tensor
            angle = angle.detach().cpu().numpy()
        if hasattr(barycenter, 'detach'):
            barycenter = barycenter.detach().cpu().numpy()

        angle      = float(np.squeeze(angle))
        barycenter = np.array(barycenter).reshape(3)  # ensure (3,)

        x_arr = np.array(x)                    # (B, T, J, 3), safe copy
        x_inv = self._rotate_xy(x_arr, -angle) # --- Inverse rotation: apply -angle ---
        x_inv = x_inv + barycenter             # Re-add barycenter

        # --- Restore original NaN/zero mask ---
        # Positions that were NaN/missing before transform were zeroed;
        # set them back to NaN so downstream code handles them correctly.
        nan_mask = np.all(x_arr == 0, axis=-1)  # (B, T, J) — all coords zero
        x_inv[nan_mask] = np.nan

        return x_inv

    # ------------------------------------------------------------------ #
    #  __call__                                                            #
    # ------------------------------------------------------------------ #
    def __call__(self, x, *args, x_supp=(), **kwargs):
        """
        Args:   x:      (T, J, 3) primary sequence
                x_supp: tuple of supplementary sequences, same shape
        Returns:     x_prime:      (T, J, 3) view-invariant primary sequence
                     x_supp_prime: tuple of transformed supplementary sequences
                    kwargs:       updated with VI_barycenter, VI_angle,
                                  min_sample, max_sample
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




class NormalizeCube:
    """
    Per-sample normalization to [-1, 1] using a SINGLE scale factor across all axes 
    — preserves the aspect ratio / shape of the skeleton (fits in a cube).
    """
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        return 'Normalize_Cube'
    
    
    @staticmethod
    def _compute_cube_params(min_, max_):
        """
        Compute center and amplitude for cube normalization.
        Args    :min_: (3,) per-axis minimum, max_: (3,) per-axis maximum
        Returns:
            center:    (3,) midpoint per axis
            amplitude: scalar — largest range across all axes
        """
        center    = (max_ + min_) / 2              # (3,) per-axis center
        amplitude = np.max(max_ - min_)            # scalar — largest range wins
        return center, amplitude

    @staticmethod
    def _normalize(x, center, amplitude):
        """
        Apply cube normalization.
        Safe against zero amplitude (returns 0 for degenerate sequences).
        Args:
            x:         (..., 3)
            center:    (3,)
            amplitude: scalar
        Returns:
            normalized array, same shape as x
        """
        if amplitude == 0:
            return np.zeros_like(x)
        return 2 * (x - center) / amplitude

    @staticmethod
    def _unnormalize(x, center, amplitude):
        """
        Inverse of cube normalization.

        Args:
            x:         (..., 3)
            center:    (3,)
            amplitude: scalar
        Returns:
            reconstructed array, same shape as x
        """
        return amplitude / 2 * x + center

    
    def __call__(self, x, *args, x_supp=(), **kwargs):
        """
        Forward normalization.
        Args:   x:          (T, J, 3)
                x_supp:     tuple of supplementary sequences, same shape
        Returns:    x_prime:      (T, J, 3) normalized, all axes in [-1, 1]
                    x_supp_prime: tuple of normalized supplementary sequences
                    kwargs:       updated with min_sample (3,), max_sample (3,)
        """
        max_ = np.nanmax(x, axis=(0, 1))  # should be of shape 3 (for the x, y, and z axes)
        min_ = np.nanmin(x, axis=(0, 1))  # same

        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print(f'[Problem in NormalizeCube] {min_}, {max_}, {x}')
        kwargs['min_sample'] = min_
        kwargs['max_sample'] = max_
        
        center, amplitude = self._compute_cube_params(min_, max_)
        # center (3,) and amplitude scalar broadcast naturally over (T, J, 3)
        x_prime = self._normalize(x, center, amplitude)
        x_supp_prime = tuple(self._normalize(xx, center, amplitude) for xx in x_supp)

        return x_prime, x_supp_prime, kwargs




    def untransform(self, x, *args, **kwargs):
        if len(x.shape) not in (3, 4):
        raise ValueError(f"[NormalizeCube.untransform] Expected 3D (T,J,3) or 4D "
                         f"(B,T,J,3) input, got shape {x.shape}")


        min_sample = kwargs['min_sample']
        max_sample = kwargs['max_sample']

        # Handle torch tensors
        if hasattr(min_sample, 'detach'):
            min_sample = min_sample.detach().cpu().numpy()
        if hasattr(max_sample, 'detach'):
            max_sample = max_sample.detach().cpu().numpy()

        min_sample = np.array(min_sample)
        max_sample = np.array(max_sample)

        # --- Reshape for broadcasting over (... T, J, 3) ---
        # (3,)   → (      3,) broadcasts over (T, J, 3) and (B, T, J, 3) naturally
        # (B, 3) → (B, 1, 1, 3) broadcasts over (B, T, J, 3)
        if min_sample.ndim == 2:                      # (B, 3) → (B, 1, 1, 3)
            min_sample = min_sample[:, None, None, :]
            max_sample = max_sample[:, None, None, :]
            amplitude  = np.max(max_sample - min_sample, axis=-1, keepdims=True) # (B, 1, 1, 1)
        else:                                         # (3,) scalar amplitude
            amplitude  = np.max(max_sample - min_sample)

        center = (max_sample + min_sample) / 2        # matches x shape via broadcast

        return self._unnormalize(np.array(x), center, amplitude)



class Normalize:
    """
    Per-sample normalization to [-1, 1] independently for each axis (X, Y, Z).
    Min/max are computed from the primary sequence x and applied consistently
    to all supplementary sequences x_supp.
    """

    def __init__(self, **kwargs):
        # No parent init needed — standalone class
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
        # Avoid division by zero: where range is 0, output 0 (midpoint of [-1,1])
        safe_range = np.where(range_ == 0, 1.0, range_)
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
        Args:
            x:    (..., 3)
            min_: (3,)
            max_: (3,)
        Returns:
            reconstructed array, same shape as x
        """
        range_ = max_ - min_                          # (3,)
        return min_ + range_ * (1 + x) / 2

    def __call__(self, x, *args, x_supp=(), **kwargs):
        """
        Forward normalization.
        Args: x:      (T, J, 3)
              x_supp: tuple of supplementary sequences, same shape
        Returns:    x_prime:      (T, J, 3) normalized to [-1, 1]
                    x_supp_prime: tuple of normalized supplementary sequences
        kwargs:       updated with min_sample (3,) and max_sample (3,)
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

    def untransform(self, x, *args, **kwargs):
        """
        Inverse normalization: map [-1, 1] back to original coordinate range.
        Args:   x:      (T, J, 3) or (B, T, J, 3)
                kwargs: must contain 'min_sample' (3,) and 'max_sample' (3,)
                    as numpy arrays or torch tensors
        Returns: reconstructed array, same shape as x
        """
        min_sample = kwargs['min_sample']
        max_sample = kwargs['max_sample']

        # Handle torch tensors
        if hasattr(min_sample, 'detach'):
            min_sample = min_sample.detach().cpu().numpy()
        if hasattr(max_sample, 'detach'):
            max_sample = max_sample.detach().cpu().numpy()

        min_sample = np.array(min_sample).reshape(3)  # ensure (3,)
        max_sample = np.array(max_sample).reshape(3)

        if len(x.shape) not in (3, 4):
            raise ValueError(f"[Normalize.untransform] Expected 3D (T,J,3) or 4D (B,T,J,3) "
                             f"input, got shape {x.shape}")

        # min_sample shape (3,) broadcasts naturally over (..., 3)
        return self._unnormalize(np.array(x), min_sample, max_sample)
