"""
Plot MABe22 Fly Trajectory Data

This script provides functions to visualize fly poses and trajectories from the MABe22 dataset.

Data format: keypoints shape = (n_frames, n_flies, 24_keypoints, 2_coords)

Keypoint structure (0-indexed):
  0-18: Physical body parts with x,y pixel coordinates
    - Wings (0-1), eyes (3-4), body parts (2, 5-8), legs (9-18)
  19-23: Derived features with paired values
    - 19: Ellipse center (x, y coordinates)
    - 20: Ellipse orientation (cos, sin) 
    - 21: Ellipse axes (major, minor lengths)
    - 22: Body areas (body area, foreground area)
    - 23: Appearance (contrast, min neighbor distance)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors

# MABe22 Fly keypoint names (24 keypoints)
# From the actual keypoint_vocabulary in the dataset
KEYPOINT_NAMES = [
    'wing_left',           # 0
    'wing_right',          # 1
    'antennae',            # 2
    'right_eye',           # 3
    'left_eye',            # 4
    'left_shoulder',       # 5 (left front of thorax)
    'right_shoulder',      # 6 (right front of thorax)
    'end_notum',           # 7 (base/end of thorax)
    'end_abdomen',         # 8 (tip of abdomen)
    'middle_left_b',       # 9 (left middle femur base)
    'middle_left_e',       # 10 (left middle femur-tibia joint)
    'middle_right_b',      # 11 (right middle femur base)
    'middle_right_e',      # 12 (right middle femur-tibia joint)
    'tip_front_right',     # 13 (right front leg tip)
    'tip_middle_right',    # 14 (right middle leg tip)
    'tip_back_right',      # 15 (right rear leg tip)
    'tip_back_left',       # 16 (left rear leg tip)
    'tip_middle_left',     # 17 (left middle leg tip)
    'tip_front_left',      # 18 (left front leg tip)
    'ellipse_center',      # 19 (body center x, y)
    'ellipse_orientation', # 20 (cos_ori, sin_ori)
    'ellipse_axes',        # 21 (major axis, minor axis)
    'body_areas',          # 22 (body area, foreground area)
    'appearance',          # 23 (image contrast, min foreground distance)
]

# Define skeleton connections from AnimalPoseForecasting repository
# These are the verified connections used in their research
SKELETON_EDGES = np.array([
    [7, 8],    # base_thorax to tip_abdomen
    [10, 14],  # left_middle_femur_tibia_joint to tip_middle_right
    [11, 12],  # middle_right_b to middle_right_e
    [12, 17],  # middle_right_e to tip_middle_left
    [7, 11],   # base_thorax to middle_right_b
    [9, 10],   # middle_left_b to middle_left_e
    [7, 9],    # base_thorax to middle_left_b
    [5, 7],    # left_shoulder to base_thorax
    [2, 3],    # antennae to right_eye
    [2, 7],    # antennae to base_thorax
    [5, 18],   # left_shoulder to tip_front_left
    [6, 13],   # right_shoulder to tip_front_right
    [7, 16],   # base_thorax to tip_back_left
    [7, 15],   # base_thorax to tip_back_right
    [2, 4],    # antennae to left_eye
    [6, 7],    # right_shoulder to base_thorax
    [7, 0],    # base_thorax to wing_left
    [7, 1],    # base_thorax to wing_right
])

# Note: Keypoints 0-18 are actual body parts with x,y coordinates
# Keypoints 19-23 are derived features (not used in skeleton):
#   19: ellipse_center - body center position (x, y)
#   20: ellipse_orientation - body orientation (cos, sin)
#   21: ellipse_axes - ellipse fit (major axis, minor axis lengths)
#   22: body_areas - area measurements (body area, foreground area)
#   23: appearance - appearance metrics (contrast, neighbor distance)

# Skeleton edges are from the AnimalPoseForecasting repository
# They work with your data since the keypoint ordering (0-18) is the same

# Arena parameters from AnimalPoseForecasting config
ARENA_RADIUS_MM = 26.689  # Arena radius in millimeters
ARENA_RADIUS_PX = 507.611429  # Median arena radius in pixels over all videos

# Default arena radius - use MM since MABe22 data appears to be in mm coordinates
ARENA_RADIUS_DEFAULT = ARENA_RADIUS_MM  # Use mm by default


def plot_arena(ax=None, radius=None, center=(0, 0), 
               color='gray', linestyle='--', linewidth=2, alpha=0.5):
    """
    Plot the circular arena boundary.
    
    Args:
        ax: matplotlib axis (uses current axis if None)
        radius: arena radius (if None, uses ARENA_RADIUS_DEFAULT = 26.689 mm)
        center: (x, y) center of arena
        color: color of arena circle
        linestyle: line style for arena boundary
        linewidth: line width
        alpha: transparency
        
    Returns:
        circle: matplotlib Circle artist
    """
    if ax is None:
        ax = plt.gca()
    
    if radius is None:
        radius = ARENA_RADIUS_DEFAULT
    
    circle = plt.Circle(center, radius, fill=False, 
                       color=color, linestyle=linestyle, 
                       linewidth=linewidth, alpha=alpha, zorder=1)
    ax.add_artist(circle)
    
    return circle


def get_Dark3_cmap():
    """
    Create an extended Dark2 colormap with more colors.
    Returns Dark2 colors followed by darker versions of the same colors.
    """
    dark2 = list(cm.get_cmap('Dark2').colors)
    dark3 = dark2.copy()
    for c in dark2:
        chsv = colors.rgb_to_hsv(c)
        chsv[2] = chsv[2] / 2.
        crgb = colors.hsv_to_rgb(chsv)
        dark3.append(crgb)
    dark3cm = colors.ListedColormap(tuple(dark3))
    return dark3cm


def is_real_fly(pose):
    """
    Check if a fly is real (not NaN padding).
    
    Args:
        pose: array of shape (n_keypoints, 2) - single fly pose
        
    Returns:
        bool: True if fly has enough valid (non-NaN) data
    """
    # Check if at least some keypoints are valid (not all NaN)
    # A fly is real if at least 50% of keypoints are not NaN
    n_valid = np.sum(~np.isnan(pose[:, 0]))
    return n_valid > (len(pose) * 0.5)


def plot_fly(pose, ax=None, kptcolors='hsv', skelcolor=[0.6, 0.6, 0.6], 
             plotskel=True, plotkpts=True, kpt_ms=6, skel_lw=1,
             kpt_alpha=1., skel_alpha=1., kpt_marker='.', name=None,
             only_physical_keypoints=True):
    """
    Plot a single fly's pose with keypoints and skeleton.
    
    Args:
        pose: array of shape (n_keypoints, 2) - x,y coordinates of keypoints
        ax: matplotlib axis (created if None)
        kptcolors: colormap or color for keypoints
        skelcolor: color for skeleton edges
        plotskel: whether to plot skeleton
        plotkpts: whether to plot keypoints
        kpt_ms: keypoint marker size
        skel_lw: skeleton line width
        kpt_alpha: keypoint transparency
        skel_alpha: skeleton transparency
        kpt_marker: marker style for keypoints
        name: label for this fly
        only_physical_keypoints: if True, only plot keypoints 0-18 (body parts), 
                                 not 19-23 (derived features)
        
    Returns:
        fig, ax: matplotlib figure and axis handles
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Check if this fly is real (not NaN padding)
    if not is_real_fly(pose):
        return fig, ax
    
    # Select which keypoints to plot
    if only_physical_keypoints and pose.shape[0] >= 19:
        pose_to_plot = pose[:19]  # Only physical body parts
    else:
        pose_to_plot = pose
    
    # Plot keypoints
    if plotkpts:
        xc = pose_to_plot[:, 0]
        yc = pose_to_plot[:, 1]
        
        # Filter out NaN keypoints
        valid_mask = ~np.isnan(xc) & ~np.isnan(yc)
        xc_valid = xc[valid_mask]
        yc_valid = yc[valid_mask]
        
        if len(xc_valid) > 0:
            # Check if kptcolors is a string
            if isinstance(kptcolors, str):
                # Try to use it as a colormap first
                try:
                    kptcolors_map = plt.get_cmap(kptcolors)
                    # It's a valid colormap
                    ax.scatter(xc_valid, yc_valid, c=np.arange(len(xc_valid)), marker=kpt_marker, 
                              cmap=kptcolors_map, s=kpt_ms**2, alpha=kpt_alpha, zorder=10)
                except (ValueError, KeyError):
                    # Not a colormap, treat as a color name
                    kptname = 'keypoints'
                    if name is not None:
                        kptname = f'{name} {kptname}'
                    ax.plot(xc_valid, yc_valid, kpt_marker, color=kptcolors, label=kptname, 
                           zorder=10, ms=kpt_ms, alpha=kpt_alpha, mew=0)
            else:
                # Use single color (list/tuple/array)
                kptname = 'keypoints'
                if name is not None:
                    kptname = f'{name} {kptname}'
                ax.plot(xc_valid, yc_valid, kpt_marker, color=kptcolors, label=kptname, 
                       zorder=10, ms=kpt_ms, alpha=kpt_alpha, mew=0)
    
    # Plot skeleton
    if plotskel:
        segments = pose[SKELETON_EDGES, :]
        
        # Filter out segments that have NaN values
        valid_segments = []
        for segment in segments:
            # Only include segment if both endpoints are valid (not NaN)
            if not (np.isnan(segment[0, 0]) or np.isnan(segment[0, 1]) or 
                    np.isnan(segment[1, 0]) or np.isnan(segment[1, 1])):
                valid_segments.append(segment)
        
        if len(valid_segments) > 0:
            line_collection = matplotlib.collections.LineCollection(
                valid_segments, colors=skelcolor, linewidths=skel_lw, alpha=skel_alpha
            )
            ax.add_collection(line_collection)
    
    return fig, ax


def plot_flies(poses, ax=None, colors=None, kptcolors=[0, 0, 0], 
               plotskel=True, plotkpts=True, textlabels=None, 
               plot_arena_boundary=True, **kwargs):
    """
    Plot multiple flies for a single frame.
    
    Args:
        poses: array of shape (n_keypoints, 2, n_flies) - poses for all flies
        ax: matplotlib axis (created if None)
        colors: colormap or list of colors for each fly's skeleton
        kptcolors: color for keypoints (same for all flies)
        plotskel: whether to plot skeletons
        plotkpts: whether to plot keypoints
        textlabels: if not None, add text labels ('keypoints' or fly IDs)
        plot_arena_boundary: whether to plot the circular arena boundary
        **kwargs: additional arguments passed to plot_fly
        
    Returns:
        fig, ax: matplotlib figure and axis handles
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    # Plot arena boundary first (so it's behind the flies)
    if plot_arena_boundary:
        plot_arena(ax=ax)
    
    # Set default colormap for flies
    if colors is None:
        colors = get_Dark3_cmap()
    
    n_flies = poses.shape[-1]
    
    # Convert colormap to color array if needed
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors)
        colors = cmap(np.linspace(0., 1., n_flies))
    elif hasattr(colors, '__call__'):  # if it's a colormap object
        colors = colors(np.linspace(0., 1., n_flies))
    
    # Plot each fly
    for fly_idx in range(n_flies):
        pose = poses[:, :, fly_idx]
        
        # Skip if this fly is NaN (padding)
        if not is_real_fly(pose):
            continue
        
        # Set color for this fly
        if isinstance(colors, np.ndarray) and colors.ndim == 2:
            skelcolor = colors[fly_idx]
        else:
            skelcolor = colors
        
        # Set text label
        label = None
        if textlabels == 'fly_ids':
            label = f'Fly {fly_idx}'
        elif textlabels == 'keypoints':
            label = 'keypoints'
        
        # Plot this fly
        plot_fly(pose, ax=ax, skelcolor=skelcolor, kptcolors=kptcolors,
                plotskel=plotskel, plotkpts=plotkpts, name=label, **kwargs)
    
    ax.axis('equal')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    
    return fig, ax


def plot_frame(data, sequence_id, frame_idx, ax=None, **kwargs):
    """
    Plot all flies for a specific frame from a sequence.
    
    Args:
        data: loaded .npy file (with 'sequences' key)
        sequence_id: string ID of the sequence
        frame_idx: frame number to plot (0 to 4499)
        ax: matplotlib axis (created if None)
        **kwargs: additional arguments passed to plot_flies
        
    Returns:
        fig, ax: matplotlib figure and axis handles
    """
    # Extract keypoints for this frame
    # Shape: (n_flies, n_keypoints, 2) -> need to transpose to (n_keypoints, 2, n_flies)
    keypoints = data['sequences'][sequence_id]['keypoints'][frame_idx]  # (11, 24, 2)
    
    # Transpose to expected format: (24, 2, 11)
    poses = np.transpose(keypoints, (1, 2, 0))
    
    # Plot
    fig, ax = plot_flies(poses, ax=ax, **kwargs)
    
    if ax is None:
        ax = plt.gca()
    ax.set_title(f'Sequence: {sequence_id}, Frame: {frame_idx}')
    
    return fig, ax


def plot_trajectory(data, sequence_id, fly_idx=0, frames=None, 
                   keypoint_idx=19, ax=None, color='blue', **kwargs):
    """
    Plot the trajectory of a specific keypoint for a specific fly over time.
    
    Args:
        data: loaded .npy file (with 'sequences' key)
        sequence_id: string ID of the sequence
        fly_idx: which fly to plot (0-10)
        frames: list/array of frame indices (if None, plot all frames)
        keypoint_idx: which keypoint to track (default 19 = ellipse_center/body center)
        ax: matplotlib axis (created if None)
        color: color for trajectory line
        **kwargs: additional arguments for plot
        
    Returns:
        fig, ax: matplotlib figure and axis handles
    """
    keypoints = data['sequences'][sequence_id]['keypoints']  # (4500, 11, 24, 2)
    
    if frames is None:
        frames = np.arange(keypoints.shape[0])
    
    # Extract trajectory: (n_frames, 2)
    trajectory = keypoints[frames, fly_idx, keypoint_idx, :]
    
    # Remove NaN values
    valid = ~np.isnan(trajectory[:, 0])
    trajectory = trajectory[valid]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, 
           label=f'Fly {fly_idx}, {KEYPOINT_NAMES[keypoint_idx]}', **kwargs)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', ms=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', ms=10, label='End')
    
    ax.axis('equal')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_title(f'Trajectory: Sequence {sequence_id}')
    ax.legend()
    
    return fig, ax


# Example usage
if __name__ == '__main__':
    print("MABe22 Fly Plotting Module")
    print(f"Number of keypoints: {len(KEYPOINT_NAMES)}")
    print(f"Number of skeleton edges: {len(SKELETON_EDGES)}")
    print("\nKeypoint names:")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {i}: {name}")
    
    print("\nTo use this module:")
    print("1. Load your data: data = np.load('fly_group_train.npy', allow_pickle=True).item()")
    print("2. Plot a frame: plot_frame(data, sequence_id='01FJRKCP4GE1W1DFX51C', frame_idx=0)")
    print("3. Plot trajectory: plot_trajectory(data, sequence_id='01FJRKCP4GE1W1DFX51C', fly_idx=0)")