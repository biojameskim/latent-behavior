import numpy as np
from pathlib import Path

# Load full training data
train_data = np.load('../../../data/fly_data/fly_group_train.npy', allow_pickle=True).item()

# Parameters for subset
N_SEQUENCES = 50
FRAMES_PER_SEQ = 300
FLY_INDEX = 0  # Just use first fly from each sequence

# Prepare subset
subset_data = {
    'trajectories': [],
    'labels': [],
    'sequence_ids': [],
    'keypoint_vocabulary': train_data['keypoint_vocabulary'],
    'label_vocabulary': train_data['vocabulary']
}

# Sample random sequences
sequence_ids = list(train_data['sequences'].keys())
np.random.seed(42)
selected_ids = np.random.choice(sequence_ids, size=N_SEQUENCES, replace=False)

for seq_id in selected_ids:
    seq_data = train_data['sequences'][seq_id]
    
    # Get keypoints: shape (4500, 11, 24, 2)
    keypoints = seq_data['keypoints']
    
    # Take first 300 frames, first fly
    trajectory = keypoints[:FRAMES_PER_SEQ, FLY_INDEX, :, :]  # (300, 24, 2)
    
    # Flatten to (300, 48) - easier to work with
    trajectory_flat = trajectory.reshape(FRAMES_PER_SEQ, -1)
    
    # Get annotations: shape (3, 4500)
    annotations = seq_data['annotations'][:, :FRAMES_PER_SEQ]  # (3, 300)
    
    # For simplicity, just use first task as labels
    labels = annotations[0, :]  # (300,)
    
    subset_data['trajectories'].append(trajectory_flat)
    subset_data['labels'].append(labels)
    subset_data['sequence_ids'].append(seq_id)

# Convert to arrays
subset_data['trajectories'] = np.array(subset_data['trajectories'])  # (50, 300, 48)
subset_data['labels'] = np.array(subset_data['labels'])  # (50, 300)

# Add metadata
subset_data['metadata'] = {
    'n_sequences': N_SEQUENCES,
    'frames_per_sequence': FRAMES_PER_SEQ,
    'frame_rate': 30,
    'n_keypoints': 24,
    'n_coordinates': 2,
    'description': 'Subset of MABe22 fly trajectories for discovery agent testing',
    'fly_index_used': FLY_INDEX,
    'task_used': train_data['vocabulary'][0]  # 'control'
}

# Save
np.savez_compressed('../../../data/fly_data/mabe22_subset_for_claude.npz', **subset_data)

print(f"Created subset with:")
print(f"  Sequences: {subset_data['trajectories'].shape[0]}")
print(f"  Frames per sequence: {subset_data['trajectories'].shape[1]}")
print(f"  Features per frame: {subset_data['trajectories'].shape[2]}")
print(f"  Total size: {subset_data['trajectories'].nbytes / 1024 / 1024:.2f} MB")
print(f"  Labels available: {np.isfinite(subset_data['labels']).sum()} / {subset_data['labels'].size} frames")


### Expected Output:
# Created subset with:
#   Sequences: 50
#   Frames per sequence: 300
#   Features per frame: 48
#   Total size: 0.57 MB
#   Labels available: 15000 / 15000 frames