"""
Generate synthetic behavioral trajectories with known ground truth structure.

This provides a controlled testbed for validating that discovery agents can
rediscover known behavioral modes using only intrinsic metrics.

Key features:
- Known ground truth (which behaviors exist and when they occur)
- Controllable difficulty (noise level, mode overlap, dimensionality)
- Unlimited data generation for ablations
- Same structure as real behavior data (trajectories → discrete modes)

Use cases:
1. Validate that intrinsic metrics correlate with rediscovery
2. Test discovery agents before applying to real data
3. Ablation studies (vary noise, complexity, etc.)
4. Quick iteration during development
"""

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class SyntheticBehaviorGenerator:
    """
    Generate synthetic trajectories with distinct behavioral modes.

    Behavioral modes are defined as different motion patterns in a 2D space
    (e.g., circling, straight lines, random walk, oscillation, etc.).
    """

    def __init__(
        self,
        n_modes: int = 5,
        n_keypoints: int = 3,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            n_modes: Number of distinct behavioral modes
            n_keypoints: Number of keypoints to track (like fly body parts)
            random_seed: Random seed for reproducibility
        """
        self.n_modes = n_modes
        self.n_keypoints = n_keypoints

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_mode_trajectory(
        self,
        mode_id: int,
        duration: int,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Generate trajectory for a single behavioral mode.

        Args:
            mode_id: Which behavior mode (0 to n_modes-1)
            duration: Number of frames
            noise_level: Gaussian noise std

        Returns:
            trajectory: (duration, n_keypoints * 2) array
        """
        t = np.arange(duration)

        # Define distinct motion patterns for each mode
        if mode_id == 0:
            # Circular motion (clockwise)
            radius = 1.0
            theta = 2 * np.pi * t / 50
            base_x = radius * np.cos(theta)
            base_y = radius * np.sin(theta)

        elif mode_id == 1:
            # Linear motion (right)
            speed = 0.05
            base_x = speed * t
            base_y = np.zeros_like(t)

        elif mode_id == 2:
            # Oscillation (vertical)
            freq = 0.1
            base_x = np.zeros_like(t)
            base_y = np.sin(2 * np.pi * freq * t)

        elif mode_id == 3:
            # Figure-8 pattern
            freq = 0.05
            base_x = np.sin(2 * np.pi * freq * t)
            base_y = np.sin(4 * np.pi * freq * t)

        elif mode_id == 4:
            # Random walk (Brownian motion)
            steps_x = np.random.randn(duration) * 0.1
            steps_y = np.random.randn(duration) * 0.1
            base_x = np.cumsum(steps_x)
            base_y = np.cumsum(steps_y)

        else:
            # For additional modes, create variations
            mode_variant = mode_id % 5
            variation = (mode_id // 5) + 1

            # Recursively generate base pattern and modify
            base_traj = self.generate_mode_trajectory(mode_variant, duration, noise_level=0)
            base_x = base_traj[:, 0] * variation
            base_y = base_traj[:, 1] * variation

        # Create trajectory for all keypoints
        # Keypoints move together but with small offsets
        trajectory = np.zeros((duration, self.n_keypoints * 2))

        for kp_idx in range(self.n_keypoints):
            # Small spatial offset for each keypoint
            offset_x = (kp_idx - self.n_keypoints / 2) * 0.1
            offset_y = (kp_idx - self.n_keypoints / 2) * 0.05

            # x coordinates
            trajectory[:, kp_idx * 2] = base_x + offset_x
            # y coordinates
            trajectory[:, kp_idx * 2 + 1] = base_y + offset_y

        # Add noise
        if noise_level > 0:
            noise = np.random.randn(*trajectory.shape) * noise_level
            trajectory += noise

        return trajectory

    def generate_sequence(
        self,
        total_frames: int = 1000,
        mode_probs: Optional[np.ndarray] = None,
        mean_bout_length: int = 50,
        noise_level: float = 0.1,
        transition_smoothing: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full trajectory sequence with mode switches.

        Args:
            total_frames: Total sequence length
            mode_probs: Probability of each mode (if None, uniform)
            mean_bout_length: Average duration of each behavioral bout
            noise_level: Gaussian noise std
            transition_smoothing: Frames to smooth at mode transitions

        Returns:
            trajectory: (total_frames, n_keypoints * 2) full trajectory
            labels: (total_frames,) ground truth mode labels
        """
        if mode_probs is None:
            mode_probs = np.ones(self.n_modes) / self.n_modes

        trajectory = np.zeros((total_frames, self.n_keypoints * 2))
        labels = np.zeros(total_frames, dtype=int)

        current_frame = 0

        while current_frame < total_frames:
            # Sample mode and bout duration
            mode = np.random.choice(self.n_modes, p=mode_probs)
            bout_length = int(np.random.exponential(mean_bout_length)) + 1

            # Clip to remaining frames
            bout_length = min(bout_length, total_frames - current_frame)

            # Generate trajectory for this bout
            bout_traj = self.generate_mode_trajectory(
                mode_id=mode,
                duration=bout_length,
                noise_level=noise_level,
            )

            # Smooth transition from previous mode
            if current_frame > 0 and transition_smoothing > 0:
                smooth_frames = min(transition_smoothing, bout_length)
                weights = np.linspace(0, 1, smooth_frames)
                prev_position = trajectory[current_frame - 1]

                for i in range(smooth_frames):
                    bout_traj[i] = (1 - weights[i]) * prev_position + weights[i] * bout_traj[i]

            # Add to full trajectory
            trajectory[current_frame:current_frame + bout_length] = bout_traj
            labels[current_frame:current_frame + bout_length] = mode

            current_frame += bout_length

        return trajectory, labels

    def generate_dataset(
        self,
        n_sequences: int = 10,
        sequence_length: int = 1000,
        noise_level: float = 0.1,
        mean_bout_length: int = 50,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate multiple trajectory sequences (like multiple flies/trials).

        Args:
            n_sequences: Number of independent sequences
            sequence_length: Frames per sequence
            noise_level: Gaussian noise std
            mean_bout_length: Average bout duration

        Returns:
            trajectories: List of (seq_len, n_features) arrays
            labels: List of (seq_len,) ground truth labels
        """
        trajectories = []
        labels_list = []

        for i in range(n_sequences):
            traj, lbls = self.generate_sequence(
                total_frames=sequence_length,
                noise_level=noise_level,
                mean_bout_length=mean_bout_length,
            )
            trajectories.append(traj)
            labels_list.append(lbls)

        return trajectories, labels_list

    def visualize_modes(self, duration: int = 200, save_path: Optional[str] = None):
        """
        Visualize all behavioral modes side-by-side.
        """
        fig, axes = plt.subplots(1, self.n_modes, figsize=(4 * self.n_modes, 4))

        if self.n_modes == 1:
            axes = [axes]

        for mode_id in range(self.n_modes):
            traj = self.generate_mode_trajectory(mode_id, duration, noise_level=0.05)

            # Plot trajectory (just first keypoint for clarity)
            x = traj[:, 0]
            y = traj[:, 1]

            axes[mode_id].plot(x, y, alpha=0.7, linewidth=2)
            axes[mode_id].scatter(x[0], y[0], c='green', s=100, label='Start', zorder=5)
            axes[mode_id].scatter(x[-1], y[-1], c='red', s=100, label='End', zorder=5)
            axes[mode_id].set_title(f'Mode {mode_id}')
            axes[mode_id].set_xlabel('X')
            axes[mode_id].set_ylabel('Y')
            axes[mode_id].legend()
            axes[mode_id].grid(True, alpha=0.3)
            axes[mode_id].axis('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved mode visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_sequence(
        self,
        trajectory: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Visualize a sequence with ground truth labels.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot trajectory colored by mode
        x = trajectory[:, 0]
        y = trajectory[:, 1]

        for mode_id in range(self.n_modes):
            mask = labels == mode_id
            axes[0].scatter(x[mask], y[mask], s=5, label=f'Mode {mode_id}', alpha=0.6)

        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Trajectory colored by ground truth mode')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')

        # Plot mode labels over time
        axes[1].plot(labels, linewidth=2)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Mode')
        axes[1].set_title('Ground truth mode over time')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-0.5, self.n_modes - 0.5])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved sequence visualization to {save_path}")
        else:
            plt.show()

        plt.close()


def generate_difficulty_sweep(
    base_generator: SyntheticBehaviorGenerator,
    noise_levels: List[float] = [0.05, 0.1, 0.2, 0.3],
    n_sequences: int = 10,
) -> dict:
    """
    Generate datasets with varying difficulty levels.

    Useful for testing how discovery agents perform as task gets harder.
    """
    datasets = {}

    for noise in noise_levels:
        print(f"Generating dataset with noise={noise:.2f}")
        trajs, lbls = base_generator.generate_dataset(
            n_sequences=n_sequences,
            noise_level=noise,
        )
        datasets[f'noise_{noise:.2f}'] = {
            'trajectories': trajs,
            'labels': lbls,
            'noise_level': noise,
        }

    return datasets


if __name__ == '__main__':
    print("Synthetic Behavior Data Generator")
    print("=" * 80)

    # Create generator
    generator = SyntheticBehaviorGenerator(
        n_modes=5,
        n_keypoints=3,
        random_seed=42,
    )

    # Visualize individual modes
    print("\n1. Visualizing individual behavioral modes...")
    generator.visualize_modes(duration=200, save_path='synthetic_modes.png')

    # Generate a sequence
    print("\n2. Generating example sequence...")
    trajectory, labels = generator.generate_sequence(
        total_frames=1000,
        noise_level=0.1,
        mean_bout_length=50,
    )

    print(f"   Trajectory shape: {trajectory.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Unique modes: {np.unique(labels)}")

    # Compute some statistics
    bout_lengths = []
    current_bout = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            current_bout += 1
        else:
            bout_lengths.append(current_bout)
            current_bout = 1

    print(f"   Mean bout length: {np.mean(bout_lengths):.1f} frames")
    print(f"   Number of bouts: {len(bout_lengths)}")

    # Visualize sequence
    print("\n3. Visualizing sequence...")
    generator.visualize_sequence(trajectory, labels, save_path='synthetic_sequence.png')

    # Generate full dataset
    print("\n4. Generating full dataset (10 sequences)...")
    trajectories, labels_list = generator.generate_dataset(
        n_sequences=10,
        sequence_length=1000,
        noise_level=0.1,
    )

    print(f"   Generated {len(trajectories)} sequences")
    print(f"   Total frames: {sum(len(t) for t in trajectories)}")

    # Save dataset
    print("\n5. Saving dataset...")
    np.savez(
        'synthetic_dataset.npz',
        trajectories=trajectories,
        labels=labels_list,
        n_modes=generator.n_modes,
        n_keypoints=generator.n_keypoints,
    )
    print("   Saved to synthetic_dataset.npz")

    print("\n" + "=" * 80)
    print("Done! You can now use this synthetic data to test discovery agents.")
    print("\nNext steps:")
    print("1. Apply your VQ-VAE to this data")
    print("2. Evaluate with discovery_metrics.py")
    print("3. Check: Do intrinsic metrics correlate with rediscovery ARI?")
