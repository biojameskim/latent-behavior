"""
PyTorch Dataset for fly keypoint trajectories.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import logging

LOG = logging.getLogger(__name__)


class FlyKeypointDataset(Dataset):
    """
    Dataset that creates sliding windows from fly trajectories for VQ-VAE training.
    """
    
    def __init__(self, all_fly_trajectories, window_size=150, stride=75, include_metadata=False):
        """
        Args:
            all_fly_trajectories: list of dicts from preprocessing.load_and_preprocess_for_vqvae
            window_size: number of frames per window
            stride: stride for sliding window
        """
        self.window_size = window_size
        self.stride = stride
        self.include_metadata = include_metadata
        self.windows = []
        self.metadata = []
        
        LOG.info(f"Creating windows with size={window_size}, stride={stride}")
        
        for traj_dict in all_fly_trajectories:
            keypoints = traj_dict['keypoints']  # (4500, 24, 2)
            windows = self._create_windows(keypoints)
            
            if windows is not None:
                for i, window in enumerate(windows):
                    self.windows.append(window)
                    self.metadata.append({
                        'sequence_id': traj_dict['sequence_id'],
                        'fly_idx': traj_dict['fly_idx'],
                        'window_idx': i
                    })
        
        # Convert to tensor: (N, window_size, 24, 2) -> (N, window_size, 48) -> (N, 48, window_size)
        self.windows = torch.FloatTensor(np.array(self.windows))
        self.windows = self.windows.reshape(
            self.windows.shape[0],
            self.windows.shape[1],
            -1
        )
        # Transpose to (N, features, time) for Conv1d
        self.windows = self.windows.permute(0, 2, 1)

        LOG.info(f"Created dataset with {len(self.windows)} windows")
        LOG.info(f"Window shape: {self.windows.shape} (batch, features, time)")
    
    def _create_windows(self, keypoints):
        """Create sliding windows from a single trajectory."""
        num_frames = keypoints.shape[0]
        
        if num_frames < self.window_size:
            return None
        
        windows = []
        for start in range(0, num_frames - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = keypoints[start:end]
            windows.append(window)
        
        return np.array(windows) if len(windows) > 0 else None
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Returns a single window of shape (48, window_size) = (features, time)"""
        window = self.windows[idx]
        if self.include_metadata:
            return window, self.metadata[idx]
        return window
    
    def get_metadata(self, idx):
        """Get metadata for a specific window."""
        return self.metadata[idx]


def create_dataloaders(train_data_file, val_data_file=None, 
                       window_size=150, stride=75, 
                       batch_size=32, num_workers=4,
                       train_fly_ids=None, val_fly_ids=None,
                       train_include_metadata=False, val_include_metadata=False):
    """
    Convenience function to create train/val dataloaders.
    
    Args:
        train_data_file: path to training .npy file
        val_data_file: optional path to validation .npy file
        window_size: window size for Dataset
        stride: stride for Dataset
        batch_size: batch size for DataLoader
        num_workers: number of workers for DataLoader
    
    Returns:
        train_loader, val_loader (or just train_loader if no val file)
    """
    from torch.utils.data import DataLoader
    from .preprocessing import load_and_preprocess_for_vqvae
    
    # Load training data
    if train_fly_ids is not None:
        train_fly_ids = set(train_fly_ids)
        LOG.info("Building training set with %d specified flies", len(train_fly_ids))
    train_trajectories = load_and_preprocess_for_vqvae(
        train_data_file,
        allowed_fly_ids=train_fly_ids,
    )
    train_dataset = FlyKeypointDataset(
        train_trajectories,
        window_size,
        stride,
        include_metadata=train_include_metadata,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if val_fly_ids is not None and val_data_file is None:
        LOG.info("No validation data file provided; reusing training data file for validation split")
        val_data_file = train_data_file
    
    if val_data_file is not None:
        if val_fly_ids is not None:
            val_fly_ids = set(val_fly_ids)
            LOG.info("Building validation set with %d specified flies", len(val_fly_ids))
        val_trajectories = load_and_preprocess_for_vqvae(
            val_data_file,
            allowed_fly_ids=val_fly_ids,
        )
        val_dataset = FlyKeypointDataset(
            val_trajectories,
            window_size,
            stride,
            include_metadata=val_include_metadata,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader
    
    return train_loader
