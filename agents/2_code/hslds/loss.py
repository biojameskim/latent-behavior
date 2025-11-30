"""
Discovery-specific loss function
Universal across all architectures
"""

import torch
import torch.nn.functional as F
import numpy as np


def discovery_loss(original, reconstructed, codes, model, alpha=1.0, beta=0.25, gamma=0.1, delta=0.5,
                   epsilon=0.01, zeta=0.5):
    """
    Multi-objective loss for unsupervised discovery.
    This loss function MUST be used regardless of architecture to ensure fair comparison.

    Components:
    1. Reconstruction (alpha): Can we decode back?
    2. Commitment (beta): Stable codes during training?
    3. Codebook utilization (gamma): All codes used?
    4. Temporal coherence (delta): Codes persist over time?
    5. Entropy bonus (epsilon): Encourage diverse state usage (FIX #3)
    6. State diversity (zeta): Ensure different states produce different outputs (FIX #4)

    Args:
        original: (Batch, Time, Features) input data
        reconstructed: (Batch, Time, Features) reconstructed data
        codes: (Batch, Time) discrete codes
        model: DiscoveryPipeline instance
        alpha, beta, gamma, delta, epsilon, zeta: loss weights

    Returns:
        total_loss: scalar tensor
        loss_dict: dictionary of individual loss components
    """
    # 1. Reconstruction
    recon_loss = F.mse_loss(reconstructed, original)

    # 2. Commitment (Adaptive: if model has it, use it, else 0)
    if hasattr(model, 'commitment_loss'):
        commit_loss = model.commitment_loss
    else:
        commit_loss = torch.tensor(0.0).to(original.device)

    # 3. Codebook utilization (maximize entropy of code usage)
    # Flattens batch/time dimensions
    codes_flat = codes.flatten()
    n_codes = model.n_codes

    # Count code usage
    code_counts = torch.bincount(codes_flat, minlength=n_codes)
    code_probs = code_counts.float() / codes_flat.numel()
    code_probs = code_probs + 1e-10  # numerical stability

    # Compute entropy
    codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))

    # We want to MAXIMIZE entropy, so we minimize negative entropy
    target_entropy = np.log(n_codes)
    codebook_loss = target_entropy - codebook_entropy

    # 4. Temporal coherence (Penalize rapid flickering)
    # Calculate probability of code switching frame-to-frame
    code_changes = (codes[:, 1:] != codes[:, :-1]).float().mean()
    temporal_loss = code_changes

    # 5. FIX #3: Entropy bonus (encourage diverse state usage in each batch)
    # This is different from codebook_loss - it encourages per-batch diversity
    # to prevent early convergence to single state
    batch_code_counts = torch.bincount(codes_flat, minlength=n_codes).float()
    batch_code_probs = batch_code_counts / codes_flat.numel()
    batch_code_probs = batch_code_probs + 1e-10  # numerical stability

    # Maximize entropy (minimize negative entropy)
    batch_entropy = -torch.sum(batch_code_probs * torch.log(batch_code_probs))
    max_entropy = np.log(n_codes)
    entropy_bonus_loss = max_entropy - batch_entropy  # minimize this to maximize entropy

    # 6. FIX #4: State diversity loss (ensure different states produce different outputs)
    # Sample different states and ensure their embeddings are different
    # This prevents collapse where all states produce similar outputs
    if hasattr(model.decoder, 'state_embed'):
        # Get all state embeddings
        all_state_ids = torch.arange(n_codes, device=original.device)
        state_embeds = model.decoder.state_embed(all_state_ids)  # (n_codes, hidden_dim)

        # Compute pairwise cosine similarity
        state_embeds_norm = F.normalize(state_embeds, p=2, dim=1)
        similarity_matrix = torch.mm(state_embeds_norm, state_embeds_norm.t())  # (n_codes, n_codes)

        # We want OFF-diagonal elements to be small (states should be different)
        # Create mask to exclude diagonal
        mask = ~torch.eye(n_codes, device=original.device, dtype=torch.bool)
        off_diagonal_sim = similarity_matrix[mask]

        # Penalize high similarity between different states
        diversity_loss = off_diagonal_sim.abs().mean()
    else:
        diversity_loss = torch.tensor(0.0).to(original.device)

    # Combined
    total_loss = (
        alpha * recon_loss +
        beta * commit_loss +
        gamma * codebook_loss +
        delta * temporal_loss +
        epsilon * entropy_bonus_loss +
        zeta * diversity_loss
    )

    return total_loss, {
        'total': total_loss.item(),
        'reconstruction': recon_loss.item(),
        'commitment': commit_loss.item(),
        'codebook': codebook_loss.item(),
        'temporal': temporal_loss.item(),
        'entropy_bonus': entropy_bonus_loss.item(),
        'diversity': diversity_loss.item()
    }
