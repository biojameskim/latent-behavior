"""
Data preprocessing utilities for fly tracking data.

Extracts from prepare_data.py the core preprocessing logic:
- Loads multi-fly sequences
- Splits into individual fly trajectories
- Removes flies with NaN values
"""

import logging
from typing import Optional, Set, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

FlyID = Tuple[str, int]


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

    Args:
        data_file (str): Path to .npy file containing fly tracking data

    Returns:
        list of dict: Each dict contains:
            - 'keypoints': (n_frames, 24, 2) array of keypoints for this fly
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
                # Perfect tracking - keep entire trajectory
                all_fly_trajectories.append({
                    'keypoints': fly_keypoints,  # (4500, 24, 2) - guaranteed no NaNs
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
