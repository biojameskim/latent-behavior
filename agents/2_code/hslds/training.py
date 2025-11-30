"""
Training loop for DiscoveryPipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from loss import discovery_loss


class FailureModeDetector:
    """Detect and warn about common failure modes"""

    def check_all(self, codes, original, reconstructed, n_codes):
        """Run all checks"""
        issues = []
        if self.check_codebook_collapse(codes, n_codes):
            issues.append("CODEBOOK_COLLAPSE")
        if self.check_temporal_flickering(codes):
            issues.append("TEMPORAL_FLICKERING")
        if self.check_degenerate_segmentation(codes):
            issues.append("DEGENERATE_SEGMENTATION")
        if self.check_poor_reconstruction(original, reconstructed):
            issues.append("POOR_RECONSTRUCTION")
        return issues

    def check_codebook_collapse(self, codes, n_codes):
        unique = len(np.unique(codes.cpu().numpy()))
        return unique < n_codes * 0.10  # Strict threshold

    def check_temporal_flickering(self, codes):
        change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
        return change_rate > 0.5

    def check_degenerate_segmentation(self, codes):
        freqs = np.bincount(codes.flatten().cpu().numpy()) / codes.numel()
        return freqs.max() > 0.9  # One code dominates 90% of data

    def check_poor_reconstruction(self, original, reconstructed):
        # simple check: is variance of recon << variance of original?
        var_orig = torch.var(original)
        var_recon = torch.var(reconstructed)
        return var_recon < (var_orig * 0.1)  # Posterior collapse to mean


def train_pipeline(model, data, epochs=50, batch_size=32, lr=1e-3, device='cuda'):
    """
    Standard PyTorch training loop calling discovery_loss

    Args:
        model: DiscoveryPipeline instance
        data: torch.Tensor (N, Time, Features) - full dataset
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        device: 'cuda' or 'cpu'

    Returns:
        trained model, history dict
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    failure_detector = FailureModeDetector()

    # Dataset
    n_samples = data.shape[0]
    indices = np.arange(n_samples)

    history = {
        'total_loss': [],
        'reconstruction': [],
        'commitment': [],
        'codebook': [],
        'temporal': [],
        'entropy_bonus': [],
        'diversity': [],
        'failures': []
    }

    print(f"Training on {n_samples} sequences for {epochs} epochs")
    print(f"Device: {device}")

    for epoch in range(epochs):
        # Update model's current epoch for temperature annealing
        model.current_epoch = epoch

        epoch_losses = {
            'total': [],
            'reconstruction': [],
            'commitment': [],
            'codebook': [],
            'temporal': [],
            'entropy_bonus': [],
            'diversity': []
        }

        # Shuffle data
        np.random.shuffle(indices)

        # Mini-batch training
        num_batches = (n_samples + batch_size - 1) // batch_size
        pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Get batch
            batch = data[batch_indices].to(device)

            # Preprocess
            batch_processed = model.preprocessor.preprocess(batch)

            # Forward pass
            reconstructed, codes = model(batch_processed)

            # Compute loss
            loss, loss_dict = discovery_loss(
                batch_processed,
                reconstructed,
                codes,
                model,
                alpha=1.0,
                beta=0.5,     # INCREASED from 0.25 - stronger commitment
                gamma=0.0,    # REMOVED from 0.1 - let model find natural # of states
                delta=1.0,    # INCREASED from 0.5 - penalize flickering more
                epsilon=0.01, # FIX #3: entropy bonus to encourage diverse state usage
                zeta=0.5      # FIX #4: state diversity loss to prevent collapse
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            for key in epoch_losses.keys():
                epoch_losses[key].append(loss_dict[key])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}"
            })

        # Epoch statistics
        for key in epoch_losses.keys():
            avg_loss = np.mean(epoch_losses[key])
            if key == 'total':
                history['total_loss'].append(avg_loss)
            else:
                history[key].append(avg_loss)

        # Check for failure modes at end of epoch
        with torch.no_grad():
            model.eval()
            # Check on first batch
            sample_batch = data[:batch_size].to(device)
            sample_processed = model.preprocessor.preprocess(sample_batch)
            sample_recon, sample_codes = model(sample_processed)

            failures = failure_detector.check_all(
                sample_codes,
                sample_processed,
                sample_recon,
                model.n_codes
            )
            history['failures'].append(failures)

            if failures:
                print(f"\n[WARNING] Failure modes detected: {failures}")

            model.train()

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {history['total_loss'][-1]:.4f}, "
              f"Recon: {history['reconstruction'][-1]:.4f}, "
              f"Codebook: {history['codebook'][-1]:.4f}, "
              f"Temporal: {history['temporal'][-1]:.4f}")

    return model, history
