import numpy as np

"""
Skeleton normalization: centering + scale normalization + rotation normalization.
Input shape convention: (N, T, V, 3)
Root joint = mean of (left_shoulder, right_shoulder, left_hip, right_hip),

Because z is the true vertical axis, rotation normalization here is YAW-ONLY:  we rotate about the z-axis to 
align the body's facing direction (shoulder line) to a canonical direction in the xy-plane. We deliberately do NOT 
do a full 3D re-orientation (e.g. aligning a spine vector to an axis) -- that would tilt the skeleton and 
corrupt the vertical axis, which is already meaningful/canonical in your coordinate system.
"""

class NormalizeConfig:
    def __init__(self, left_shoulder: int=6, right_shoulder: int=9, left_hip: int=12, right_hip: int=15, eps: float=1e-8):
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder
        self.left_hip = left_hip
        self.right_hip = right_hip
        self.eps = eps


def _compute_root(x: np.ndarray, cfg: NormalizeConfig) -> np.ndarray:
    # x: (T, V, 3) -> returns (T, 3)
    joints = x[:, [cfg.left_shoulder, cfg.right_shoulder, cfg.left_hip, cfg.right_hip], :]   # (T, 4, 3)
    return joints.mean(axis=1)


"""
def center_skeleton(x: np.ndarray, cfg: NormalizeConfig = NormalizeConfig()) -> np.ndarray:
    #Subtract the computed root (per-frame) from every joint.
    root = _compute_root(x, cfg)[:, None, :]   # (T, 1, 3)
    return x - root


def scale_normalize(x: np.ndarray, cfg: NormalizeConfig = NormalizeConfig()) -> np.ndarray:
    #Normalize by average torso size: distance between mid-shoulder and mid-hip, averaged across all valid frames per sequence.
    mid_shoulder = (x[:, :, cfg.left_shoulder, :] + x[:, :, cfg.right_shoulder, :]) / 2.0
    mid_hip = (x[:, :, cfg.left_hip, :] + x[:, :, cfg.right_hip, :]) / 2.0
    torso_len = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)   # (N, T)

    valid = torso_len > cfg.eps
    scale = np.zeros(x.shape[0], dtype=x.dtype)
    for n in range(x.shape[0]):
        if valid[n].any():
            scale[n] = torso_len[n][valid[n]].mean()
        else:
            scale[n] = 1.0  # fallback: no-op if sequence is degenerate

    scale = np.maximum(scale, cfg.eps).reshape(-1, 1, 1, 1)
    return x / scale
"""


def _yaw_rotation_matrix(left_sh, right_sh, eps):
    """
    Compute a rotation-about-z (yaw) matrix per sequence that aligns the shoulder line (left_shoulder -> right_shoulder), 
    projected onto the xy-plane, to the canonical +x axis. z is left completely unchanged.

    left_sh, right_sh: (N, 3) -- one reference frame per sequence.
    Returns R: (N, 3, 3).
    """
    shoulder_vec = right_sh - left_sh                           # (3, )
    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0]) 

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.zeros((3, 3), dtype=shoulder_vec.dtype)
    # Rotate by -angle about z so shoulder_vec lands on +x axis:
    R[0, 0] = cos_a
    R[0, 1] = sin_a
    R[1, 0] = -sin_a
    R[1, 1] = cos_a
    R[2, 2] = 1.0
    return R



def rotation_normalize(x: np.ndarray, cfg: NormalizeConfig = NormalizeConfig(), ref_frame: str = "first") -> np.ndarray:
    """
    Rotate every frame about the z-axis (yaw only) so the body's facing direction is canonical. z is untouched, only x,y rotate.
    ref_frame: "first" uses frame 0 to compute the rotation angle. 
                "mean"  averages joint positions across all frames first (more stable if frame 0 is noisy / partially occluded).
    x: (T, V, 3) -> returns same shape.
    """
    if ref_frame == "first":
        ref = x[0, :, :]        # (V, 3)
    elif ref_frame == "mean":
        ref = x.mean(axis=0)    # (V, 3)
    else:
        raise ValueError("ref_frame must be 'first' or 'mean'")
    
    left_sh = ref[cfg.left_shoulder, :]
    right_sh = ref[cfg.right_shoulder, :]
    R = _yaw_rotation_matrix(left_sh, right_sh, cfg.eps)   # (3, 3)
    # Apply the same yaw rotation to every frame/joint of the sequence.
    x_rot = np.einsum('ij,tvj->tvi', R, x)

    return x_rot






#ESSENTIAL_JOINTS = [1, 3, 6, 8]
ESSENTIAL_JOINTS = [3, 4, 5, 6, 9, 12, 15]

def compute_svd(points):
    """
    points:     (n_keypoints, 3) numpy array. Should be left right hip, coord and back to approximate the plane of the mouse back.
    :returns:   barycenter:         (3,) numpy array — mean position of valid points
                transition matrix: (3, 3), both can be in transform_points 
    """
    hip_coord_back = np.full(points.shape, np.nan)
    for i in ESSENTIAL_JOINTS:
        hip_coord_back[i,:] = points[i,:]
    points = hip_coord_back[~np.any(np.isnan(hip_coord_back), axis=1)] # Remove rows with missing values
    if len(points) == 0:
        return np.nan, np.nan
    barycenter = np.nanmean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - barycenter) # center the data then apply SVD
    
    return barycenter, Vt.T




class ViewInvariant:
    """
    Applies a rotation in the XY plane (optionally XZ plane for pitch) to make skeleton sequences view-invariant. No norm transformation is applied.
    - Compute SVD on a reference frame to find the body's principal axes.
        - For standing/walking: use A[:,0] (spine axis) — has large XY component.
        - For climbing:         use A[:,2] (perpendicular to back) — spine is vertical so A[:,0] has near-zero XY component → unstable.
    - Climbing detected when the spine axis (A[:,0]) is dominated by Z.
    - After rotation: body axis aligned with +X (y=0 plane).
    - Facing direction (±X ambiguity) resolved using left/right hip joints.

    Forward pass  (T, J, 3) or (B, T, J, 3):  __call__   → centers + rotates by +angle
    Inverse pass  (B, T, J, 3):                untransform → rotates by -angle + re-adds barycenter
    """
    def __init__(self, index_frame=0, left_idx=None, right_idx=None, if_rotate_xz=False, **kwargs):
        self.index_frame = index_frame
        self.left_idx = left_idx   # e.g., left hip
        self.right_idx = right_idx # e.g., right hip 
        self.if_rotate_xz = if_rotate_xz

    def __str__(self):
        return 'ViewInvariant'

    @staticmethod
    def _rotate_xy(array, angle):
        """
        Apply a 2D rotation of `angle` radians in the XY plane.
        Args:   array: (..., 3), angle: float
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        out = np.empty_like(array)
        x = array[..., 0]
        y = array[..., 1]
        out[..., 0] = x * cos_a - y * sin_a
        out[..., 1] = x * sin_a + y * cos_a
        out[..., 2] = array[..., 2] # Z remains unchanged
        return out
    
    @staticmethod
    def _rotate_xz(array, pitch):
        cos_a, sin_a = np.cos(pitch), np.sin(pitch)
        out = np.empty_like(array)
        x = array[..., 0]
        z = array[..., 2]
        out[..., 0] = x * cos_a + z * sin_a
        out[..., 1] = array[..., 1]
        out[..., 2] = - x * sin_a + z * cos_a # Z remains unchanged
        return out
        

    def _needs_flip(self, rotated_points, A, angle, index_vect=0):
        """
        Check if the mouse is facing -X after rotation, needs a 180° flip. Uses left/right hip joints: forward = cross(left→right, spine).
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
        spine = A[:, index_vect].copy()
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
        valid_essential = np.array([np.sum([not np.any(np.isnan(x[t, j]))for j in ESSENTIAL_JOINTS]) for t in range(x.shape[0])]) 
        if np.max(valid_essential) == 0:
            raise ValueError("[ViewInvariant] No frame found where at least one of joints is valid.")
        
        # 1. Pick reference frame
        idx    = int(np.argmax(valid_essential))     # frame with most valid essentials
        points = x[idx]                              # (J, 3)
        mask_na = np.any(np.isnan(points), axis=1)   # checks in reference frame, per joint, whether any of its 3 coordinates is NaN.
        n_valid = np.sum(~mask_na)
        if n_valid < 2:     # Final guard: need at least 2 valid joints for SVD to be meaningful
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
        
        # 5. Calculate Pitch = angle between spine axis and XY plane. 
        spine_axis = A[:, 0] # Record spine axis 
        pitch = np.arcsin(np.clip(spine_axis[2], -1.0, 1.0)) # arcsin(Z component) gives how far the spine tilts out of XY

        # 6. Check and fix facing direction using left/right hips
        rotated   = self._rotate_xy(points - barycenter, angle)
        if self._needs_flip(rotated, A, angle, index_vect):
            angle += np.pi  # absorb 180° flip into the angle

        return barycenter, index_vect, angle, pitch


    def apply_transform(self, x, barycenter, angle, pitch):
        """
        Forward transform: center + rotate by +angle in XY.
        Args:   x:(T, J, 3) or None; barycenter: (3,); angle:float
        """
        if x is None:
            return None
        out = ViewInvariant._rotate_xy(x - barycenter, angle)
        if self.if_rotate_xz:
            return ViewInvariant._rotate_xz(out, -pitch)
        return out

   
    def untransform(self, x, **kwargs):
        """
        Inverse transform: rotate by -angle + re-add barycenter.
        Args:    x:      (B, T, J, 3) — batched sequences in canonical frame
            kwargs:      must contain 'VI_angle' and 'VI_barycenter'
                     (scalars, numpy arrays, or torch tensors)
        Returns:
            (B, T, J, 3) sequences restored to original coordinate frame
        """
        angle      = kwargs['VI_angle']
        barycenter = kwargs['VI_barycenter']
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
        barycenter, index_vect, angle, pitch = self.compute_transform(x)
        x_prime = self.apply_transform(x, barycenter, angle, pitch)

        x_supp_prime = tuple(self.apply_transform(xx, barycenter, angle) for xx in x_supp)
        if np.all(np.isnan(x_prime)):
            print('[ViewInvariant] Warning: all NaN in x_prime')

        kwargs['VI_barycenter'] = barycenter
        kwargs['VI_angle']      = angle
        kwargs['min_sample']    = np.nanmin(x_prime, axis=(0, 1))  # (3,)
        kwargs['max_sample']    = np.nanmax(x_prime, axis=(0, 1))  # (3,)

        return x_prime, x_supp_prime, kwargs
