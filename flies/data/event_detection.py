"""
Rigorous behavioral event detection for fly trajectories.

This module provides validated event detection methods for identifying
behavioral events like walking onset, turning, and stopping.

Usage:
    from flies.data.event_detection import BehaviorEventDetector

    detector = BehaviorEventDetector(
        keypoint_names=['body_center', 'wing_l', ...],
        fps=30.0,
        pixel_to_mm=0.1
    )

    # Detect walking onsets
    walking_starts = detector.detect_walking_onset(
        keypoints,
        speed_threshold_mm_s=5.0,
        min_duration_s=0.2
    )
"""

import numpy as np
from typing import List, Optional, Tuple


class BehaviorEventDetector:
    """
    Detect behavioral events in fly trajectories.

    This class provides methods to detect specific behavioral events
    (walking, turning, stopping) using physically meaningful thresholds.
    """

    def __init__(
        self,
        keypoint_names: Optional[List[str]] = None,
        fps: float = 30.0,
        pixel_to_mm: float = 0.1,
        body_keypoint: str = 'body_center'
    ):
        """
        Initialize event detector.

        Args:
            keypoint_names: List of keypoint names in order. If None, assumes
                          body center is at index 0.
            fps: Frames per second of video
            pixel_to_mm: Conversion factor from pixels to millimeters
            body_keypoint: Name of keypoint to use for body position
        """
        self.keypoint_names = keypoint_names or []
        self.fps = fps
        self.pixel_to_mm = pixel_to_mm

        # Find body center index
        if keypoint_names and body_keypoint in keypoint_names:
            self.body_idx = keypoint_names.index(body_keypoint)
        else:
            self.body_idx = 0  # Default to first keypoint

    def detect_walking_onset(
        self,
        keypoints: np.ndarray,
        speed_threshold_mm_s: float = 5.0,
        min_duration_s: float = 0.2,
        min_gap_s: float = 0.5,
        smoothing_window_s: float = 0.1
    ) -> np.ndarray:
        """
        Detect walking onset events with physical units.

        Walking is defined as sustained movement above a speed threshold.
        This method uses smoothing to reduce noise and requires minimum
        duration and gap between walking bouts.

        Args:
            keypoints: Keypoint array, shape (T, num_keypoints, 2) or (T, 2)
            speed_threshold_mm_s: Minimum speed to count as walking (mm/s)
            min_duration_s: Minimum bout duration (seconds)
            min_gap_s: Minimum gap between bouts (seconds)
            smoothing_window_s: Temporal smoothing window (seconds)

        Returns:
            onset_frames: (N,) array of frame indices where walking starts

        Example:
            >>> detector = BehaviorEventDetector(fps=30, pixel_to_mm=0.1)
            >>> keypoints = np.random.randn(1000, 24, 2)
            >>> onsets = detector.detect_walking_onset(keypoints)
            >>> print(f"Found {len(onsets)} walking bouts")
        """
        # Extract body positions
        if keypoints.ndim == 3:
            positions = keypoints[:, self.body_idx, :]  # (T, 2)
        elif keypoints.ndim == 2:
            positions = keypoints  # Already (T, 2)
        else:
            raise ValueError(f"Expected 2D or 3D keypoints, got shape {keypoints.shape}")

        # Convert to physical units (mm)
        positions_mm = positions * self.pixel_to_mm

        # Compute speed in mm/s
        dt = 1.0 / self.fps
        velocity = np.diff(positions_mm, axis=0) / dt  # (T-1, 2)
        speed = np.linalg.norm(velocity, axis=1)  # (T-1,)

        # Pad to original length
        speed = np.concatenate([[0], speed])  # (T,)

        # Smooth speed to reduce noise
        window_frames = int(smoothing_window_s * self.fps)
        if window_frames > 0:
            kernel = np.ones(window_frames) / window_frames
            speed = np.convolve(speed, kernel, mode='same')

        # Detect walking periods
        walking = speed > speed_threshold_mm_s

        # Find bout starts and ends
        walking_padded = np.concatenate([[False], walking, [False]])
        bout_starts = np.where(np.diff(walking_padded.astype(int)) == 1)[0]
        bout_ends = np.where(np.diff(walking_padded.astype(int)) == -1)[0]

        # Filter by minimum duration
        min_duration_frames = int(min_duration_s * self.fps)
        durations = bout_ends - bout_starts
        valid_duration = durations >= min_duration_frames

        bout_starts = bout_starts[valid_duration]
        bout_ends = bout_ends[valid_duration]

        # Filter by minimum gap between bouts
        if len(bout_starts) > 1:
            min_gap_frames = int(min_gap_s * self.fps)
            gaps = bout_starts[1:] - bout_ends[:-1]

            # Keep first bout, then only those with sufficient gap
            keep_mask = np.concatenate([[True], gaps >= min_gap_frames])
            bout_starts = bout_starts[keep_mask]

        return bout_starts

    def detect_turning(
        self,
        keypoints: np.ndarray,
        angle_threshold_deg: float = 45.0,
        window_frames: int = 5,
        min_gap_s: float = 0.3,
        min_speed_mm_s: float = 2.0
    ) -> np.ndarray:
        """
        Detect sharp turns in trajectory.

        Turns are detected when the heading direction changes by more than
        a threshold angle. Requires minimum speed to exclude stationary
        orientation changes.

        Args:
            keypoints: Keypoint array, shape (T, num_keypoints, 2) or (T, 2)
            angle_threshold_deg: Minimum angle change (degrees)
            window_frames: Window over which to compute angle change
            min_gap_s: Minimum gap between turn events (seconds)
            min_speed_mm_s: Minimum speed to count as turn (mm/s)

        Returns:
            turn_frames: (N,) array of frame indices where turns occur
        """
        # Extract positions
        if keypoints.ndim == 3:
            positions = keypoints[:, self.body_idx, :]
        else:
            positions = keypoints

        # Convert to mm
        positions_mm = positions * self.pixel_to_mm

        # Compute speed (to filter stationary turns)
        dt = 1.0 / self.fps
        velocity = np.diff(positions_mm, axis=0) / dt
        speed = np.linalg.norm(velocity, axis=1)
        speed = np.concatenate([[0], speed])

        # Compute heading angles
        angles = np.arctan2(velocity[:, 1], velocity[:, 0])

        # Compute angular change over window
        if window_frames > len(angles):
            return np.array([], dtype=int)

        angle_diffs = np.zeros(len(angles) - window_frames)
        for i in range(len(angle_diffs)):
            # Angular difference accounting for wraparound
            diff = angles[i + window_frames] - angles[i]
            # Wrap to [-π, π]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            angle_diffs[i] = np.abs(diff)

        # Convert to degrees
        angle_diffs_deg = np.degrees(angle_diffs)

        # Find turns above threshold and with sufficient speed
        turn_candidates = np.where(angle_diffs_deg > angle_threshold_deg)[0]

        # Filter by speed at turn location
        turn_frames = []
        for idx in turn_candidates:
            turn_frame = idx + window_frames // 2  # Center of window
            if turn_frame < len(speed) and speed[turn_frame] > min_speed_mm_s:
                turn_frames.append(turn_frame)

        turn_frames = np.array(turn_frames)

        # Filter by minimum gap
        if len(turn_frames) > 1:
            min_gap_frames = int(min_gap_s * self.fps)
            gaps = np.diff(turn_frames)
            keep_mask = np.concatenate([[True], gaps >= min_gap_frames])
            turn_frames = turn_frames[keep_mask]

        return turn_frames

    def detect_stopping(
        self,
        keypoints: np.ndarray,
        speed_threshold_mm_s: float = 1.0,
        min_duration_s: float = 0.5,
        min_gap_s: float = 0.5
    ) -> np.ndarray:
        """
        Detect stopping events.

        Stopping is defined as sustained low speed. Similar to walking
        detection but with inverted threshold.

        Args:
            keypoints: Keypoint array, shape (T, num_keypoints, 2) or (T, 2)
            speed_threshold_mm_s: Maximum speed to count as stopped (mm/s)
            min_duration_s: Minimum stop duration (seconds)
            min_gap_s: Minimum gap between stop events (seconds)

        Returns:
            stop_frames: (N,) array of frame indices where stopping starts
        """
        # Extract positions
        if keypoints.ndim == 3:
            positions = keypoints[:, self.body_idx, :]
        else:
            positions = keypoints

        # Convert to mm and compute speed
        positions_mm = positions * self.pixel_to_mm
        dt = 1.0 / self.fps
        velocity = np.diff(positions_mm, axis=0) / dt
        speed = np.linalg.norm(velocity, axis=1)
        speed = np.concatenate([[0], speed])

        # Smooth
        window_frames = int(0.1 * self.fps)
        if window_frames > 0:
            kernel = np.ones(window_frames) / window_frames
            speed = np.convolve(speed, kernel, mode='same')

        # Detect stopped periods
        stopped = speed < speed_threshold_mm_s

        # Find stops
        stopped_padded = np.concatenate([[False], stopped, [False]])
        stop_starts = np.where(np.diff(stopped_padded.astype(int)) == 1)[0]
        stop_ends = np.where(np.diff(stopped_padded.astype(int)) == -1)[0]

        # Filter by duration
        min_duration_frames = int(min_duration_s * self.fps)
        durations = stop_ends - stop_starts
        valid = durations >= min_duration_frames

        stop_starts = stop_starts[valid]
        stop_ends = stop_ends[valid]

        # Filter by gap
        if len(stop_starts) > 1:
            min_gap_frames = int(min_gap_s * self.fps)
            gaps = stop_starts[1:] - stop_ends[:-1]
            keep_mask = np.concatenate([[True], gaps >= min_gap_frames])
            stop_starts = stop_starts[keep_mask]

        return stop_starts

    def validate_events(
        self,
        event_frames: np.ndarray,
        sequence_length: int,
        window_size: int,
        min_offset: int = 0
    ) -> np.ndarray:
        """
        Validate that events can be used for windowing.

        Ensures events leave room for full window and are not too close
        to sequence boundaries.

        Args:
            event_frames: (N,) array of event frame indices
            sequence_length: Total sequence length
            window_size: Window size for training
            min_offset: Minimum offset from start (optional)

        Returns:
            valid_events: (M,) array of valid event indices
        """
        valid_mask = np.ones(len(event_frames), dtype=bool)

        # Must leave room for full window
        valid_mask &= event_frames + window_size <= sequence_length

        # Must not be too close to start
        if min_offset > 0:
            valid_mask &= event_frames >= min_offset

        return event_frames[valid_mask]


def visualize_events(
    keypoints: np.ndarray,
    events: dict,
    save_path: Optional[str] = None
):
    """
    Visualize detected events on trajectory.

    Args:
        keypoints: (T, num_keypoints, 2) or (T, 2) array
        events: Dict mapping event type to frame indices
                e.g., {'walking': array([10, 50, ...]), 'turning': array([...])}
        save_path: Optional path to save figure

    Example:
        >>> detector = BehaviorEventDetector()
        >>> walking = detector.detect_walking_onset(keypoints)
        >>> turning = detector.detect_turning(keypoints)
        >>> visualize_events(keypoints, {'walking': walking, 'turning': turning})
    """
    import matplotlib.pyplot as plt

    # Extract body trajectory
    if keypoints.ndim == 3:
        trajectory = keypoints[:, 0, :]  # Body center
    else:
        trajectory = keypoints

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot trajectory
    ax = axes[0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', alpha=0.3, linewidth=1)

    # Plot events
    colors = {'walking': 'green', 'turning': 'red', 'stopping': 'blue'}
    for event_type, event_frames in events.items():
        color = colors.get(event_type, 'gray')
        for frame in event_frames:
            ax.plot(trajectory[frame, 0], trajectory[frame, 1],
                   'o', color=color, markersize=8, label=event_type)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Trajectory with Events')
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.axis('equal')

    # Plot speed over time
    ax = axes[1]
    velocity = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    ax.plot(speed, 'k-', alpha=0.5, linewidth=1)

    # Mark events
    for event_type, event_frames in events.items():
        color = colors.get(event_type, 'gray')
        for frame in event_frames:
            if frame < len(speed):
                ax.axvline(frame, color=color, alpha=0.5, linestyle='--')

    ax.set_xlabel('Frame')
    ax.set_ylabel('Speed (pixels/frame)')
    ax.set_title('Speed Over Time with Events')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
