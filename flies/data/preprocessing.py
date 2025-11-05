"""
Data preprocessing utilities for fly tracking data.

Extracts from prepare_data.py the core preprocessing logic:
- Loads multi-fly sequences
- Splits into individual fly trajectories
- Removes flies with NaN values
- Normalizes trajectories to canonical frame (centered at origin, facing upward)
"""

import logging
from typing import Optional, Set, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

FlyID = Tuple[str, int]


def normalize_trajectory(keypoints):
    """
    Normalize a fly trajectory to a canonical reference frame.

    This makes all flies start at (0, 0) and face upward (positive y direction),
    removing translation and rotation variance. This is better than rotation
    augmentation because it directly creates rotation-invariant representations.

    Args:
        keypoints: (n_frames, 24, 2) array of keypoints
            - Keypoint 19: ellipse_center (body center x, y)
            - Keypoint 20: ellipse_orientation (cos_ori, sin_ori)

    Returns:
        normalized_keypoints: (n_frames, 24, 2) array in canonical frame
    """
    keypoints = keypoints.copy()

    # Get the initial body center position (keypoint 19 at frame 0)
    initial_center = keypoints[0, 19, :].copy()  # (x, y)

    # Get the initial orientation (keypoint 20 at frame 0)
    cos_ori, sin_ori = keypoints[0, 20, :]

    # Calculate the angle of current orientation
    # (cos_ori, sin_ori) represents a unit vector in the direction the fly faces
    current_angle = np.arctan2(sin_ori, cos_ori)

    # We want the fly to face upward (0, 1), which is at angle pi/2
    # So we need to rotate by: target_angle - current_angle
    rotation_angle = np.pi / 2 - current_angle

    # Create 2D rotation matrix
    cos_rot = np.cos(rotation_angle)
    sin_rot = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_rot, -sin_rot],
        [sin_rot, cos_rot]
    ])

    # Apply transformation to all frames
    for frame_idx in range(keypoints.shape[0]):
        # Center: translate so initial body center is at (0, 0)
        keypoints[frame_idx, :, :] -= initial_center

        # Rotate: apply rotation to align fly to face upward
        # For each keypoint, apply rotation matrix
        keypoints[frame_idx, :, :] = keypoints[frame_idx, :, :] @ rotation_matrix.T

    return keypoints


def _is_allowed_fly(sequence_id: str, fly_idx: int, allowed_fly_ids: Optional[Set[FlyID]]) -> bool:
    if allowed_fly_ids is None:
        return True
    return (sequence_id, fly_idx) in allowed_fly_ids


def load_and_preprocess_for_vqvae(data_file, allowed_fly_ids: Optional[Set[FlyID]] = None):
    """
    Load and preprocess fly tracking data for VQ-VAE training.

    Processing steps:
    - Removes frames with NaNs
    - Splits multi-fly sequences into individual fly trajectories
        - Rationale: VQ-VAE will learn better and the codebook will be more
          interpretable if we learn individual syllables (what is THIS fly doing?)
          rather than group behaviors (what are ALL these flies doing?)
        - Social behavior can be analyzed post hoc by examining how social context
          affects the sequence of syllables, e.g., "Fly 1 does syllable A more
          often when near other flies doing syllable B"
    - Normalizes each trajectory to canonical frame (centered at origin, facing upward)

    Args:
        data_file (str): Path to .npy file containing fly tracking data

    Returns:
        list of dict: Each dict contains:
            - 'keypoints': (n_frames, 24, 2) array of normalized keypoints for this fly
            - 'sequence_id': original sequence id
            - 'fly_idx': index of the fly in the original multi-fly data
    """
    LOG.info(f"Loading data from {data_file}...")
    data = np.load(data_file, allow_pickle=True).item()

    LOG.info(f"Found {len(data['sequences'])} sequences")

    all_fly_trajectories = []
    total_flies = 0
    total_flies_removed = 0
    flies_with_partial_nans = 0

    for seq_id, seq_data in data['sequences'].items():
        keypoints = seq_data['keypoints']  # shape: (n_frames, 11, 24, 2)
        n_frames = keypoints.shape[0]

        # Split into individual flies
        for fly_idx in range(11):
            total_flies += 1
            fly_keypoints = keypoints[:, fly_idx, :, :]  # (n_frames, 24, 2)

            # Check if this fly has ANY NaN values anywhere
            has_any_nan = np.any(np.isnan(fly_keypoints))

            if not has_any_nan:
                if not _is_allowed_fly(seq_id, fly_idx, allowed_fly_ids):
                    continue
                # Perfect tracking - normalize to canonical frame and keep trajectory
                fly_keypoints_normalized = normalize_trajectory(fly_keypoints)
                all_fly_trajectories.append({
                    'keypoints': fly_keypoints_normalized,  # (4500, 24, 2) - normalized, no NaNs
                    'sequence_id': seq_id,
                    'fly_idx': fly_idx,
                })
            else:
                # Has at least one NaN - skip this fly
                total_flies_removed += 1

                # Count how many frames are affected (for logging)
                fly_flat = fly_keypoints.reshape(n_frames, -1)
                has_nan_per_frame = np.any(np.isnan(fly_flat), axis=1)
                n_nan_frames = np.sum(has_nan_per_frame)

                if n_nan_frames < n_frames:  # Partial NaNs
                    flies_with_partial_nans += 1
                    LOG.debug(f"Seq {seq_id}, fly {fly_idx}: has NaNs in {n_nan_frames}/{n_frames} frames, skipping")
                else:  # All NaNs
                    LOG.debug(f"Seq {seq_id}, fly {fly_idx}: all frames are NaN, skipping")

    LOG.info(f"Total flies: {total_flies} -> {len(all_fly_trajectories)} (-{total_flies_removed})")
    LOG.info(f"Removed {total_flies_removed} flies with NaN values")
    LOG.info(f"  - {flies_with_partial_nans} had partial NaNs (some frames affected)")
    LOG.info(f"  - {total_flies_removed - flies_with_partial_nans} had all NaN frames")
    if len(all_fly_trajectories) > 0:
        LOG.info(f"All kept trajectories have exactly {n_frames} frames with no NaNs")

    return all_fly_trajectories
