"""
Denormalize fly trajectories back to original arena coordinates.

After training on ego-centric (normalized) data, this module allows us to
transform reconstructions back to the original arena layout for visualization.
"""

import numpy as np
from typing import Dict, Tuple
import logging

LOG = logging.getLogger(__name__)

FlyKey = Tuple[str, int]


def compute_normalization_params(keypoints: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the normalization parameters used during preprocessing.

    This extracts the transformation that was applied to normalize a fly
    to (0,0) facing upward, so we can invert it later.

    Args:
        keypoints: (n_frames, 24, 2) original keypoints BEFORE normalization

    Returns:
        initial_center: (2,) array - the fly's center position at frame 0
        rotation_angle: float - the rotation applied to make fly face upward
    """
    # Get the initial body center position (keypoint 19 at frame 0)
    initial_center = keypoints[0, 19, :].copy()  # (x, y)

    # Get the initial orientation (keypoint 20 at frame 0)
    cos_ori, sin_ori = keypoints[0, 20, :]

    # Calculate the angle of current orientation
    current_angle = np.arctan2(sin_ori, cos_ori)

    # The rotation applied during normalization
    rotation_angle = np.pi / 2 - current_angle

    return initial_center, rotation_angle


def denormalize_trajectory(
    normalized_keypoints: np.ndarray,
    initial_center: np.ndarray,
    rotation_angle: float
) -> np.ndarray:
    """
    Undo the normalization transformation to restore original arena coordinates.

    This is the INVERSE of preprocessing.normalize_trajectory()

    Args:
        normalized_keypoints: (n_frames, 24, 2) normalized trajectory (centered, rotated)
        initial_center: (2,) the fly's original center at frame 0
        rotation_angle: rotation that was applied during normalization

    Returns:
        original_keypoints: (n_frames, 24, 2) trajectory in original arena coordinates
    """
    keypoints = normalized_keypoints.copy()

    # Create INVERSE rotation matrix (rotate by -rotation_angle)
    cos_rot = np.cos(-rotation_angle)
    sin_rot = np.sin(-rotation_angle)
    inverse_rotation = np.array([
        [cos_rot, -sin_rot],
        [sin_rot, cos_rot]
    ])

    # Apply inverse transformation to all frames
    for frame_idx in range(keypoints.shape[0]):
        # 1. Undo rotation (apply inverse rotation)
        keypoints[frame_idx, :, :] = keypoints[frame_idx, :, :] @ inverse_rotation.T

        # 2. Undo translation (add back initial center)
        keypoints[frame_idx, :, :] += initial_center

    return keypoints


def load_normalization_params_from_original(
    data_file: str,
) -> Dict[FlyKey, Tuple[np.ndarray, float]]:
    """
    Load original data and compute normalization parameters for all flies.

    Args:
        data_file: Path to original .npy file with multi-fly arena data

    Returns:
        params_dict: Mapping from (sequence_id, fly_idx) to (initial_center, rotation_angle)
    """
    LOG.info(f"Loading original data to compute normalization params: {data_file}")
    data = np.load(data_file, allow_pickle=True).item()

    params_dict = {}

    for seq_id, seq_data in data['sequences'].items():
        keypoints = seq_data['keypoints']  # (n_frames, 11, 24, 2)

        # Process each fly
        for fly_idx in range(11):
            fly_keypoints = keypoints[:, fly_idx, :, :]  # (n_frames, 24, 2)

            # Skip if this fly has any NaN values (wasn't used in training)
            if np.any(np.isnan(fly_keypoints)):
                continue

            # Compute normalization parameters
            initial_center, rotation_angle = compute_normalization_params(fly_keypoints)

            params_dict[(seq_id, fly_idx)] = (initial_center, rotation_angle)

    LOG.info(f"Computed normalization params for {len(params_dict)} flies")
    return params_dict


def denormalize_reconstructions(
    reconstructions: Dict[FlyKey, np.ndarray],
    normalization_params: Dict[FlyKey, Tuple[np.ndarray, float]]
) -> Dict[FlyKey, np.ndarray]:
    """
    Denormalize all reconstructed trajectories back to original arena coordinates.

    Args:
        reconstructions: Dict mapping (seq_id, fly_idx) -> normalized trajectory (n_frames, 24, 2)
        normalization_params: Dict mapping (seq_id, fly_idx) -> (initial_center, rotation_angle)

    Returns:
        denormalized: Dict mapping (seq_id, fly_idx) -> original arena trajectory (n_frames, 24, 2)
    """
    denormalized = {}

    for key, normalized_traj in reconstructions.items():
        if key not in normalization_params:
            LOG.warning(f"No normalization params for {key}, skipping")
            continue

        initial_center, rotation_angle = normalization_params[key]
        denormalized[key] = denormalize_trajectory(
            normalized_traj,
            initial_center,
            rotation_angle
        )

    LOG.info(f"Denormalized {len(denormalized)} trajectories to original coordinates")
    return denormalized
