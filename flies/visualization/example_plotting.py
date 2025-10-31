"""
Script showing various test cases for plotting fly keypoints and trajectories.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_mabe_flies import plot_frame, plot_trajectory, KEYPOINT_NAMES

# Define paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent.parent / "data" / "fly_data"
DATA_FILE = DATA_DIR / "fly_group_train.npy"

print("Loading data...")
data = np.load(DATA_FILE, allow_pickle=True).item()
seq_id = list(data['sequences'].keys())[0]

print(f"Plotting sequence: {seq_id}, frame: 0")

# ============================================================================
# Test 1: Plot with only physical keypoints (should remove the extra dots)
# ============================================================================
fig, ax = plot_frame(data, seq_id, frame_idx=0, only_physical_keypoints=True) # True by default
plt.savefig('test_plots/test_frame_physical_only.png', dpi=150, bbox_inches='tight')
print("Saved: test_plots/test_frame_physical_only.png (only keypoints 0-18)")
plt.close()

# ============================================================================
# Test 2: Plot with all keypoints (should show all dots)
# ============================================================================
fig, ax = plot_frame(data, seq_id, frame_idx=0, only_physical_keypoints=False)
plt.savefig('test_plots/test_frame_all_keypoints.png', dpi=150, bbox_inches='tight')
print("Saved: test_plots/test_frame_all_keypoints.png (all keypoints 0-23)")
plt.close()

# ============================================================================
# Test 3: Plot without keypoints, only skeleton
# ============================================================================
fig, ax = plot_frame(data, seq_id, frame_idx=0, plotkpts=False, plotskel=True)
plt.savefig('test_plots/test_frame_skeleton_only.png', dpi=150, bbox_inches='tight')
print("Saved: test_plots/test_frame_skeleton_only.png (skeleton only)")
plt.close()

# ============================================================================
# Test 4: Plot without skeleton, only keypoints
# ============================================================================
fig, ax = plot_frame(data, seq_id, frame_idx=0, plotkpts=True, plotskel=False)
plt.savefig('test_plots/test_frame_keypoints_only.png', dpi=150, bbox_inches='tight')
print("Saved: test_plots/test_frame_keypoints_only.png (keypoints only)")
plt.close()

# ============================================================================
# Test 5: Check how many flies are in this frame
# ============================================================================
keypoints = data['sequences'][seq_id]['keypoints'][0]
n_flies = keypoints.shape[0]
print(f"\nNumber of flies in frame 0: {n_flies}")

for fly_idx in range(n_flies):
    fly_kpts = keypoints[fly_idx]
    n_valid = np.sum(~np.isnan(fly_kpts[:, 0]))
    total_kpts = fly_kpts.shape[0]
    print(f"  Fly {fly_idx}: {n_valid}/{total_kpts} valid keypoints")

# ============================================================================
# Test 6: Plot trajectories (for first 5 flies over first 3000 frames)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 12))

# Plot trajectories for all flies
colors = plt.cm.tab10(np.arange(5))
for fly_idx in range(5):
    plot_trajectory(data, seq_id, fly_idx=fly_idx, 
                   keypoint_idx=19,  # ellipse_center (body center)
                   frames=np.arange(0, 3000),  # first 3000 frames
                   ax=ax, color=colors[fly_idx], alpha=0.7)

ax.set_title(f'Fly trajectories (first 3000 frames)')
plt.savefig('test_plots/trajectories.png', dpi=150, bbox_inches='tight')
print("Saved : test_plots/trajectories.png")
plt.close()

# ============================================================================
# Test 7: Plot trajectory with pose snapshots (**not sure if this actually works)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 14))

# Plot trajectory for one fly
fly_idx = 0
frames = np.arange(0, 2000, 10)  # Every 10th frame for first 2000 frames

# Get trajectory data
keypoints_seq = data['sequences'][seq_id]['keypoints']
trajectory = keypoints_seq[frames, fly_idx, 19, :]  # ellipse_center (body center)

# Plot trajectory line
valid = ~np.isnan(trajectory[:, 0])
ax.plot(trajectory[valid, 0], trajectory[valid, 1], '-', 
       color='blue', alpha=0.5, linewidth=2, label='Trajectory')

# Plot fly poses at specific frames
snapshot_frames = [0, 500, 1000, 1500]
for snapshot_frame in snapshot_frames:
    # Extract pose for this frame
    pose = keypoints_seq[snapshot_frame, fly_idx, :, :]  # (24, 2)
    
    # Check if fly is valid
    if not np.all(np.isnan(pose)):
        from plot_mabe_flies import plot_fly
        plot_fly(pose, ax=ax, skelcolor='red', kptcolors=[0.5, 0, 0],  # RGB for dark red
                kpt_ms=4, skel_lw=2, kpt_alpha=0.8, skel_alpha=0.8)

ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', ms=15, label='Start')
ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', ms=15, label='End')
ax.axis('equal')
ax.legend()
ax.set_title(f'Fly {fly_idx} trajectory with pose snapshots')
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
plt.savefig('test_plots/trajectory_with_poses.png', dpi=150, bbox_inches='tight')
print("Saved : test_plots/trajectory_with_poses.png")
plt.close()

# ============================================================================
# Test 8: Plot multiple frames in a grid
# ============================================================================
frames_to_plot = [0, 500, 1000, 1500, 2000, 2500]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, frame_idx in enumerate(frames_to_plot):
    plot_frame(data, seq_id, frame_idx=frame_idx, ax=axes[i])
    axes[i].set_title(f'Frame {frame_idx}')

plt.tight_layout()
plt.savefig('test_plots/multiple_frames.png', dpi=150, bbox_inches='tight')
print("Saved : test_plots/multiple_frames.png")
plt.close()

# ============================================================================
# Test 9: Create a quick animation
# ============================================================================
# Create frames for first 1500 frames
for frame_idx in range(0, 3000, 2):  # Every other frame
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_frame(data, seq_id, frame_idx=frame_idx, ax=ax)
    ax.set_title(f'Frame {frame_idx}')
    plt.savefig(f'animation_frames/frame_{frame_idx:04d}.png', dpi=100)
    plt.close()

print("Saved 1500 animation frames to animation_frames/")
print("To create a video, use ffmpeg:")
print("ffmpeg -framerate 30 -pattern_type glob -i 'animation_frames/frame_*.png' -c:v libopenh264 -pix_fmt yuv420p animation.mp4")

# ============================================================================
# Test 10: Plot specific keypoints only (leg tips in this example)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 12))

# Get frame data
frame_idx = 100
keypoints_frame = data['sequences'][seq_id]['keypoints'][frame_idx]

# Plot only leg tips for all flies
leg_tip_indices = [5, 7, 9, 11, 13, 15]  # All leg tips
colors = plt.cm.rainbow(np.arange(len(leg_tip_indices)) / len(leg_tip_indices))

for fly_idx in range(keypoints_frame.shape[0]):
    fly_keypoints = keypoints_frame[fly_idx]
    
    # Skip if NaN
    if np.all(np.isnan(fly_keypoints)):
        continue
    
    # Plot leg tips
    for i, kpt_idx in enumerate(leg_tip_indices):
        x, y = fly_keypoints[kpt_idx]
        if not np.isnan(x):
            ax.plot(x, y, 'o', color=colors[i], ms=8, 
                   label=KEYPOINT_NAMES[kpt_idx] if fly_idx == 0 else None)

ax.axis('equal')
ax.legend()
ax.set_title(f'Leg tip positions - Frame {frame_idx}')
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
plt.savefig('test_plots/leg_tips_only.png', dpi=150, bbox_inches='tight')
print("Saved : test_plots/leg_tips_only.png")
plt.close()