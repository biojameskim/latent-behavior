import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Load data
data = np.load('data/fly_data/fly_group_train.npy', allow_pickle=True).item()
sequences = data['sequences']
keypoint_names = data['keypoint_vocabulary']

# Get one sequence
seq_id = list(sequences.keys())[0]
keypoints = sequences[seq_id]['keypoints']  # (4500, 11, 24, 2)
annotations = sequences[seq_id]['annotations']  # (3, 4500)

print(f"Sequence {seq_id}")
print(f"Keypoints shape: {keypoints.shape}")
print(f"Annotations shape: {annotations.shape}")

# Check for NaN flies
num_valid_flies = []
for t in range(keypoints.shape[0]):
    valid = ~np.isnan(keypoints[t, :, 0, 0])  # Check x-coord of first keypoint
    num_valid_flies.append(valid.sum())

print(f"Number of valid flies per frame: min={min(num_valid_flies)}, max={max(num_valid_flies)}")