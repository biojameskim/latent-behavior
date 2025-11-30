Diagnosis

The model collapsed to a single code: codebook_usage = 0.03125 (≈1/32), mean_bout_length = 300 (full sequence one state), temporal loss was 0 (no switches). That is classic codebook collapse/posterior collapse.
Reconstruction is mediocre (MSE ≈ 0.98) and ARI = 0, so no meaningful segmentation emerged.
MMD/ACF are moderate but likely reflect memorized single-state dynamics rather than realistic multi-behavior generation.
Decision: OPTION A — REFINE

Root cause: loss as written rewards staying in one state (temporal term penalizes switches; zero switches => zero penalty) and the codebook entropy term is too weak to prevent collapse.

Targeted fix (mechanism change):

Penalize too few switches (enforce a minimum change rate) and strengthen codebook usage pressure.
Keep the existing terms but flip the temporal pressure from “don’t switch” to “switch at least a little”.
Update discovery_loss in repos/latent-behavior/agents/2_code/hsmm_vae/pipeline.py to:

def discovery_loss(
    original, reconstructed, codes, model,
    alpha=1.0, beta=0.25, gamma=2.0, delta=1.0,
    min_change_rate=0.05, usage_floor=0.2,
):
    recon_loss = F.mse_loss(reconstructed, original)

    commit_loss = model.commitment_loss if hasattr(model, "commitment_loss") else torch.tensor(0.0, device=original.device)

    with torch.no_grad():
        code_probs = torch.bincount(codes.flatten(), minlength=model.n_codes).float() / codes.numel()
        code_probs = code_probs + 1e-10
        codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))
        target_entropy = math.log(model.n_codes)
        usage_penalty = F.relu(usage_floor - (code_probs > 0).float().mean())  # pushes more codes to be used

    codebook_loss = target_entropy - codebook_entropy + usage_penalty

    change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
    # Penalize being below the minimum change rate; above min incurs no cost
    temporal_loss = F.relu(min_change_rate - change_rate)

    total_loss = alpha * recon_loss + beta * commit_loss + gamma * codebook_loss + delta * temporal_loss
    return total_loss, {
        "total": total_loss.item(),
        "reconstruction": recon_loss.item(),
        "commitment": commit_loss.item(),
        "codebook": codebook_loss.item(),
        "temporal": temporal_loss.item(),
    }
Rerun training with these defaults (gamma=2.0, delta=1.0, min_change_rate=0.05, usage_floor=0.2). This directly discourages single-state collapse and pushes utilization of the codebook while keeping reconstruction pressure.