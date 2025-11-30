Diagnosis

Codebook collapse persists: codebook_usage = 0.03125, mean_bout_length = 300. Temporal loss hit only the min-change hinge (0.05), but the model still chooses one state across the whole sequence. ARI = 0 confirms no segmentation.
Reconstruction is mediocre (≈0.98 MSE) and MMD/ACF are in the same range as the collapsed run, so generation is likely memorized single-state dynamics, not meaningful multi-behavior synthesis.
Root cause: the current loss still under-penalizes both low usage and low switching; the hinge penalties are too weak relative to reconstruction and commitment, so the optimum remains “stay in one code.”
Decision: OPTION A — REFINE

Force code usage and switching harder by increasing weights and floors:

def discovery_loss(
    original, reconstructed, codes, model,
    alpha=1.0, beta=0.25,
    gamma=10.0,      # much stronger codebook pressure
    delta=5.0,       # much stronger temporal pressure
    min_change_rate=0.15,  # target >=15% frame changes (~45/300)
    usage_floor=0.5,       # target at least 50% of codes active
):
    recon_loss = F.mse_loss(reconstructed, original)
    commit_loss = model.commitment_loss if hasattr(model, "commitment_loss") else torch.tensor(0.0, device=original.device)
    with torch.no_grad():
        code_probs = torch.bincount(codes.flatten(), minlength=model.n_codes).float() / codes.numel()
        code_probs = code_probs + 1e-10
        codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))
        target_entropy = math.log(model.n_codes)
        usage_penalty = F.relu(usage_floor - (code_probs > 0).float().mean())
    codebook_loss = target_entropy - codebook_entropy + usage_penalty
    change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
    temporal_loss = F.relu(min_change_rate - change_rate)
    total_loss = alpha * recon_loss + beta * commit_loss + gamma * codebook_loss + delta * temporal_loss
    return total_loss, {
        "total": total_loss.item(),
        "reconstruction": recon_loss.item(),
        "commitment": commit_loss.item(),
        "codebook": codebook_loss.item(),
        "temporal": temporal_loss.item(),
    }
Re-run with these defaults. This should make single-state solutions prohibitively costly and push the model to use more codes with meaningful switches.