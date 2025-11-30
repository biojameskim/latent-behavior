# Critical Architectural Fixes for Codebook Collapse

## Problem Diagnosis

After two training attempts, the model exhibited severe codebook collapse:
- **Symptom**: Model used only 1 state out of 6 available states
- **Temporal change rate**: 0.0 (states never switched)
- **Codebook loss**: 1.79 (exactly log(6), indicating uniform prior with no learning)

**Root Cause**: The decoder architecture had a fundamental flaw - it received BOTH continuous latent representations AND discrete states. This allowed the model to bypass the discrete bottleneck entirely by encoding all information in the continuous latent space, making the discrete states irrelevant for reconstruction.

## Fixes Applied

### Fix #1: CRITICAL ARCHITECTURAL CHANGE ✅
**File**: `model.py`

**Problem**: `LinearDynamicsDecoder` accepted both `latent` and `states`, allowing continuous bypass.

**Solution**: Created new `DiscreteStateDecoder` class that ONLY accepts discrete states:

```python
class DiscreteStateDecoder(nn.Module):
    """
    Decoder that ONLY uses discrete states (no continuous bypass)
    Forces model to encode all information into state sequence
    """
    def __init__(self, n_states, output_dim, hidden_dim=128):
        super().__init__()
        self.state_embed = nn.Embedding(n_states, hidden_dim)
        self.temporal_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Sequential(...)

    def forward(self, states):
        # Input: (Batch, Time) discrete state indices
        # Output: (Batch, Time, output_dim) reconstructed data
        x = self.state_embed(states)
        x, _ = self.temporal_decoder(x)
        return self.output_proj(x)
```

**Changes made**:
- Line 260-303: Added `DiscreteStateDecoder` class
- Line 379: Replaced decoder initialization: `self.decoder = DiscreteStateDecoder(...)`
- Line 467: Updated forward pass: `reconstructed = self.decoder(states_hard)`
- Lines 432-440: Updated decode method to call `self.decoder(codes)` directly

This forces an information bottleneck - ALL reconstruction must go through the discrete states.

---

### Fix #2: Slow Temperature Annealing ✅
**File**: `model.py`, line 461

**Problem**: Temperature annealed too quickly (50 epochs), causing premature commitment to single state.

**Solution**: Slowed annealing to 80 epochs:
```python
epoch_fraction = min(self.current_epoch / 80, 1.0)  # SLOWED from 50
```

This gives the model more time to explore different state configurations before committing.

---

### Fix #3: Entropy Bonus ✅
**File**: `loss.py`, lines 67-77

**Problem**: No explicit incentive for using diverse states within each batch.

**Solution**: Added per-batch entropy bonus:
```python
# Maximize entropy (minimize negative entropy)
batch_entropy = -torch.sum(batch_code_probs * torch.log(batch_code_probs))
max_entropy = np.log(n_codes)
entropy_bonus_loss = max_entropy - batch_entropy  # minimize to maximize entropy
```

**Weight**: `epsilon=0.01` (small but non-zero encouragement)

This prevents early convergence to a single state by explicitly rewarding diverse state usage.

---

### Fix #4: State Diversity Loss ✅
**File**: `loss.py`, lines 79-99

**Problem**: Different states could produce similar outputs, making them redundant.

**Solution**: Added state embedding diversity penalty:
```python
if hasattr(model.decoder, 'state_embed'):
    # Get all state embeddings
    all_state_ids = torch.arange(n_codes, device=original.device)
    state_embeds = model.decoder.state_embed(all_state_ids)

    # Compute pairwise cosine similarity
    state_embeds_norm = F.normalize(state_embeds, p=2, dim=1)
    similarity_matrix = torch.mm(state_embeds_norm, state_embeds_norm.t())

    # Penalize high off-diagonal similarity
    mask = ~torch.eye(n_codes, device=original.device, dtype=torch.bool)
    off_diagonal_sim = similarity_matrix[mask]
    diversity_loss = off_diagonal_sim.abs().mean()
```

**Weight**: `zeta=0.5` (moderate encouragement for different states to be distinct)

This ensures different states learn to produce meaningfully different outputs.

---

## Updated Loss Function

**File**: `training.py`, lines 123-129

New loss weights:
```python
loss, loss_dict = discovery_loss(
    batch_processed, reconstructed, codes, model,
    alpha=1.0,     # reconstruction (unchanged)
    beta=0.5,      # commitment (increased from 0.25)
    gamma=0.0,     # codebook utilization (removed - let model decide)
    delta=1.0,     # temporal coherence (increased from 0.5)
    epsilon=0.01,  # NEW: entropy bonus
    zeta=0.5       # NEW: state diversity
)
```

**Rationale**:
- `beta=0.5`: Stronger commitment to encourage stable state assignments
- `gamma=0.0`: Removed forced uniform usage (was causing conflict)
- `delta=1.0`: Stronger temporal penalty to prevent flickering
- `epsilon=0.01`: Small entropy bonus to encourage exploration
- `zeta=0.5`: Moderate diversity penalty to prevent redundant states

---

## Updated History Tracking

**File**: `training.py`, lines 72-81, 90-98

Added two new loss components to history:
```python
history = {
    'total_loss': [],
    'reconstruction': [],
    'commitment': [],
    'codebook': [],
    'temporal': [],
    'entropy_bonus': [],  # NEW
    'diversity': [],      # NEW
    'failures': []
}
```

---

## Expected Behavior After Fixes

### What should change:
1. **State usage**: Should see 4-6 states used (not just 1)
2. **Temporal change rate**: Should be 0.01-0.10 (states switch occasionally)
3. **Codebook loss**: Should decrease below 1.0 (learning meaningful codes)
4. **Diversity loss**: Should stabilize around 0.3-0.5 (states are different)
5. **Entropy bonus**: Should decrease as model uses more diverse states

### What to monitor:
- Check epoch 20, 40, 60, 80 logs for state usage patterns
- If states still collapse by epoch 40, may need to:
  - Increase `epsilon` (entropy bonus) to 0.05
  - Increase `zeta` (diversity) to 1.0
  - Further slow annealing to 100 epochs

---

## Files Modified

1. **model.py**:
   - Added `DiscreteStateDecoder` class (lines 260-303)
   - Updated `DiscoveryPipeline.__init__()` (line 379)
   - Updated `forward()` method (line 467)
   - Updated `decode()` method (lines 432-440)
   - Slowed temperature annealing (line 461)

2. **loss.py**:
   - Updated function signature with `epsilon`, `zeta` (line 11-12)
   - Added entropy bonus computation (lines 67-77)
   - Added diversity loss computation (lines 79-99)
   - Updated loss combination (lines 102-109)
   - Updated return dict with new losses (lines 111-119)

3. **training.py**:
   - Updated loss call with new weights (lines 123-129)
   - Added new loss components to history (lines 72-81)
   - Added new loss components to epoch_losses (lines 90-98)

---

## Testing the Fix

Run the training script:
```bash
sbatch /share/j_sun/jjk297/repos/latent-behavior/slurm/train_hslds.sh
```

Monitor the output file for:
- State usage counts (should see multiple states used)
- Temporal change rate (should be > 0)
- Codebook loss (should decrease)
- New metrics: entropy_bonus and diversity losses

If training succeeds, you should see:
- Discovery score > 100
- ARI > 0.4
- Multiple behavioral states discovered
- Coherent temporal segmentation
