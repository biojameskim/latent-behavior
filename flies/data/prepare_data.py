"""
This script prepares fly tracking data for training a VQ-VAE model.

Train dataset has 426 sequences
Total flies: 4686 -> 4025 (-661)
    - Removed 661 flies with NaN values
    - 47 had partial NaNs (some frames affected)
    - 614 had all NaN frames
All kept trajectories have exactly 4500 frames with no NaNs
"""


import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


FlyID = Tuple[str, int]


def create_windows(keypoints, window_size, stride):
    """
    Create sliding windows from keypoint data.

    keypoints: (num_frames, num_keypoints, num_coordinates) array --> (4500, 24, 2)
    window_size: number of frames per window
    stride: number of frames to skip between windows

    Returns: (num_windows, window_size, num_keypoints, num_coordinates) array --> (num_windows, window_size, 24, 2)
    """
    num_frames, num_keypoints, num_coordinates = keypoints.shape

    windows = []
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        window = keypoints[start:end]  # (window_size, num_keypoints, num_coordinates)
        windows.append(window)

    windows = np.stack(windows, axis=0)  # (num_windows, window_size, num_keypoints, num_coordinates)
    print(f"Created {windows.shape[0]} windows of size {window_size} with stride {stride}")

    return windows

def prepare_dataset(all_fly_trajectories, window_size=150, stride=75):
    """
    Prepare entire dataset by creating windows for each sequence.

    Returns: (N, window_size, num_keypoints, num_coordinates) array
    """
    all_windows = []
    for data_dict in all_fly_trajectories:
        keypoints = data_dict['keypoints']  # (num_frames, num_keypoints, num_coordinates)
        windows = create_windows(keypoints, window_size, stride)
        all_windows.append(windows)

    all_windows = np.concatenate(all_windows, axis=0)  # (N, window_size, num_keypoints, num_coordinates)
    print(f"Prepared dataset with total {all_windows.shape[0]} windows")

    return all_windows

def _is_allowed_fly(sequence_id: str, fly_idx: int, allowed_fly_ids: Optional[Set[FlyID]]) -> bool:
    if allowed_fly_ids is None:
        return True
    return (sequence_id, fly_idx) in allowed_fly_ids


def load_and_preprocess_for_vqvae(
    data_file: str,
    allowed_fly_ids: Optional[Set[FlyID]] = None,
):
    """
    Load and preprocess fly tracking data for VQ-VAE training.
    - Removes frames with NaNs
    - Splits multi-fly sequences into individual fly trajectories
        - My intuition here is that VQ-VAE will learn better and the codebook will be more interpretable if we 
            learn individual syllables (what is THIS fly doing?) rather than group behaviors (what are ALL these flies doing?)
            Obviously behavior is social, but we can analyze that post hoc by seeing how social context affects the sequence of syllables
            like "Fly 1 does syllable A more often when near other flies doing syllable B"
    - Returns 'fly_trajectories' which is a list of dicts, each with:
            - 'keypoints': (n_frames, 24, 2) array of keypoints for this fly
            - 'sequence_id': original sequence id
            - 'fly_idx': index of the fly in the original multi-fly data
            - 'original_frame_indices': indices of frames kept from the original sequence
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
    
    LOG.info(f"Total flies: {total_flies} -> {len(all_fly_trajectories)}")
    LOG.info(f"Removed {total_flies_removed} flies with NaN values")
    LOG.info(f"  - {flies_with_partial_nans} had partial NaNs (some frames affected)")
    LOG.info(f"  - {total_flies_removed - flies_with_partial_nans} had all NaN frames")
    LOG.info(f"All kept trajectories have exactly {n_frames} frames with no NaNs")
    
    # Note: all_fly_trajectories is a list of dicts, each with keys: 'keypoints', 'sequence_id', 'fly_idx', 'original_frame_indices'
    # each keypoints is an array of shape (4500, 24, 2) --> (n_valid_frames, num_keypoints, num_coordinates)
    return all_fly_trajectories


def generate_fly_splits(
    data_file: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, int]]]:
    """
    Create a reproducible fly-level split and optionally persist to disk.

    Returns a dict with keys 'train' and 'val', each containing list of
    {'sequence_id': str, 'fly_idx': int} mappings.
    """
    trajectories = load_and_preprocess_for_vqvae(data_file)
    fly_ids: List[FlyID] = [(traj['sequence_id'], traj['fly_idx']) for traj in trajectories]

    rng = np.random.default_rng(seed)
    rng.shuffle(fly_ids)

    n_total = len(fly_ids)
    n_val = int(round(n_total * val_fraction))
    n_val = min(n_val, n_total)
    n_train = max(0, n_total - n_val)

    train_ids = fly_ids[:n_train]
    val_ids = fly_ids[n_train:n_train + n_val]

    def _convert(items: Sequence[FlyID]) -> List[Dict[str, object]]:
        return [{'sequence_id': seq_id, 'fly_idx': int(fly_idx)} for seq_id, fly_idx in items]

    splits = {
        'train': _convert(train_ids),
        'val': _convert(val_ids),
    }

    LOG.info(
        "Created fly splits with %d total flies: train=%d, val=%d",
        n_total,
        len(train_ids),
        len(val_ids),
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)
        LOG.info("Saved fly splits to %s", output_path)

    return splits
    
if __name__ == "__main__":

    all_fly_trajectories = load_and_preprocess_for_vqvae('../../../data/fly_data/fly_group_train.npy')
    # print(all_fly_trajectories[0].keys())
    # print(all_fly_trajectories[0]['keypoints'].shape)

    windows = prepare_dataset(all_fly_trajectories, window_size=150, stride=75)
    print(f"Final dataset shape: {windows.shape}")

    # =========================================================================================================================
    # Add this to see the distribution of trajectory lengths
    # =========================================================================================================================
    # all_fly_trajectories = load_and_preprocess_for_vqvae('../../../data/fly_data/fly_group_train.npy')
    # trajectory_lengths = [traj['keypoints'].shape[0] for traj in all_fly_trajectories]
    # print(f"Min length: {min(trajectory_lengths)}")
    # print(f"Max length: {max(trajectory_lengths)}")
    # print(f"Mean length: {np.mean(trajectory_lengths):.1f}")
    # print(f"Median length: {np.median(trajectory_lengths):.1f}")
    # print(f"Number < 150 frames: {sum(1 for l in trajectory_lengths if l < 150)}")
    # print(f"Number < 4500 frames: {sum(1 for l in trajectory_lengths if l < 4500)}")

    # print([l for l in trajectory_lengths if l < 4500])
    # # print their sequence ids as well
    # for traj in all_fly_trajectories:
    #     if traj['keypoints'].shape[0] < 4500:
    #         print(f"Sequence ID: {traj['sequence_id']}, Fly Index: {traj['fly_idx']}, Length: {traj['keypoints'].shape[0]}")

    # =========================================================================================================================
    ### For example, Sequence ID: 3TGMB0K6RV54NK0OHZJQ, Fly Index: 2 had length 1 (which means that it only had one valid frame)
    ### We can load it directly to see the original data:
    # =========================================================================================================================
    # data = np.load('../../../data/fly_data/fly_group_train.npy', allow_pickle=True).item()
    # seq_data = data['sequences']['3TGMB0K6RV54NK0OHZJQ']
    # keypoints = seq_data['keypoints']  # shape: (n_frames, 11, 24, 2)
    # print(f"Original shape: {keypoints.shape}")
    # fly_keypoints = keypoints[:, 2, :, :]  # (n_frames, 24, 2)
    # print(fly_keypoints)
