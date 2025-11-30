Scientific Diagnosis & Decision
1. The Diagnosis
Critical Failure Mode Analysis
The model has catastrophically failed due to severe codebook collapse and degenerate segmentation. Here's the evidence: Temporal Trajectory of Failure:
Epochs 1-7: Initial temporal flickering (change rate >50%) - model exploring state space
Epoch 8: Flickering resolves, but codebook loss jumps from 0.97 → 1.21 (unhealthy)
Epoch 10: DEGENERATE_SEGMENTATION warning appears - one state dominates >90% of data
Epochs 19-100: CODEBOOK_COLLAPSE persistent - model using <10% of available states (likely 1-2 states out of 12)
The Death Spiral:
Codebook Loss Trajectory:
Epoch 1:  0.40 (healthy diversity)
Epoch 10: 1.95 (collapsing)
Epoch 20: 2.48 (collapsed - max entropy = log(12) = 2.48)
Epoch 100: 2.48 (still collapsed)
The codebook loss reaching exactly 2.48 (≈ log(12)) means the model achieved maximum negative entropy - it's using essentially ONE state for all data.
Trade-off Analysis: Reconstruction vs. Structure
Reconstruction (MSE):
Started: 0.96
Ended: 0.64
✓ Improved by 33%
BUT this is pyrrhic victory - the model "solved" reconstruction by:
Collapsing to a single state (no switching cost)
Making the decoder into a global mean estimator
Ignoring temporal structure entirely
Evidence:
Temporal loss: 0.79 → 0.00 (model stopped switching states at all!)
Codebook entropy: maximally collapsed
Final temporal change rate: ~0.0000 (states never change)
This is not learning behavioral structure - it's learning to be a static autoencoder with a useless discrete bottleneck.
Root Cause: Architectural Design Flaw
The HSLDS architecture has a fundamental incompatibility between:
The Graph Encoder: Processes only raw 48D positions (ignoring velocities/features)
The Loss Function: Reconstructs full 121D preprocessed features
The Switching Policy: Makes decisions based on 32D graph latent (which lacks temporal information)
The problem: The graph encoder has no temporal receptive field - it processes each frame independently. The GRU in the switching policy tries to add temporal context, but by then the encoder has already thrown away velocity information. Result: The model cannot distinguish between "fly standing still" and "fly grooming" if both have similar body poses. It learns the trivial solution: one state fits all.
2. The Decision: OPTION A - REFINE
Why This Model Is Not Scientifically Useful
Zero Behavioral Segmentation: Using 1 state out of 12 provides no scientific insight
No Temporal Dynamics: Temporal change rate of 0.0 means no state transitions ever occur
Invalid Generative Model: Cannot generate realistic behavior sequences (the generate() function failed during evaluation)
Reconstruction Without Understanding: Low MSE achieved via averaging, not structure discovery
This model cannot answer any scientific questions:
"When does grooming occur?" → Cannot segment
"What triggers transitions?" → No transitions exist
"How long do behaviors last?" → All behaviors are infinite (never switch)
3. Specific Mechanism Changes Required
Fix #1: Add Temporal Convolution to Encoder (Critical)
The graph encoder must see temporal context. Replace the single-frame processing:
# In model.py, class DiscoveryPipeline

def encode_to_latent(self, x):
    """
    Encode input to continuous latent with TEMPORAL CONTEXT
    Input: (Batch, Time, Features) - preprocessed features (121D)
    Output: (Batch, Time, latent_dim)
    """
    batch_size, seq_len, feature_dim = x.shape
    
    # Extract raw keypoint positions for graph encoder
    x_positions = x[:, :, :48]  # (B, T, 48)
    
    # NEW: Add temporal convolution BEFORE graph encoder
    # This gives each frame a receptive field of ±2 frames (150ms at 30Hz)
    x_positions = x_positions.permute(0, 2, 1)  # (B, 48, T)
    self.temporal_conv = nn.Conv1d(48, 48, kernel_size=5, padding=2, groups=48)  # Depthwise
    x_temporal = self.temporal_conv(x_positions)  # (B, 48, T)
    x_temporal = x_temporal.permute(0, 2, 1)  # (B, T, 48)
    
    # Now process with graph encoder
    x_flat = x_temporal.reshape(batch_size * seq_len, 48)
    latent_flat = self.graph_encoder(x_flat)
    latent = latent_flat.reshape(batch_size, seq_len, self.latent_dim)
    
    return latent
Rationale: Drosophila behaviors have 100-300ms timescales. A 5-frame kernel (167ms) captures this.
Fix #2: Reduce Codebook Loss Weight (Critical)
The gamma=0.1 weight is forcing the model to maximize entropy, which conflicts with finding sparse behavioral structure. Real behavior uses only 3-5 states frequently.
# In train_hslds.sh or main.py, change loss weights:

loss, loss_dict = discovery_loss(
    batch_processed,
    reconstructed,
    codes,
    model,
    alpha=1.0,
    beta=0.5,      # INCREASE commitment (was 0.25)
    gamma=0.0,     # REMOVE codebook pressure (was 0.1) 
    delta=1.0      # INCREASE temporal coherence (was 0.5)
)
Rationale:
Removing gamma lets the model find the natural number of states
Increasing delta from 0.5→1.0 penalizes flickering more heavily
Increasing beta encourages codes to stabilize
Fix #3: Add Annealing Schedule for Gumbel Temperature
The Gumbel-Softmax is using a fixed temperature=1.0, which doesn't let states crystallize.
# In model.py, forward() method, replace line with temperature:

# Start hot (explore), end cold (commit)
epoch_fraction = min(current_epoch / 50, 1.0)  # Pass epoch from training
temperature = max(1.0 - 0.9 * epoch_fraction, 0.1)  # 1.0 → 0.1

state_probs = F.gumbel_softmax(state_logits, tau=temperature, hard=True)
Rationale: Start with soft exploration, then force hard decisions.
Fix #4: Reduce Number of States
12 states is too many for 50 sequences with 3 ground-truth behaviors. Overfitting the discrete space.
# In train_hslds.sh, change:
--n_states 6  # Instead of 12
Rationale: Occam's Razor - start simple. If the model uses all 6, increase later.
Fix #5: Add State Usage Regularization (Alternative to gamma=0)
Instead of maximizing entropy blindly, penalize unused states while allowing clustering:
# In loss.py, replace codebook_loss section:

# Count state usage
code_counts = torch.bincount(codes.flatten(), minlength=n_codes)
unused_states = (code_counts == 0).sum()

# Penalize having unused states, but don't force uniform usage
codebook_loss = unused_states.float()  # Simple: count of dead codes
Rationale: Encourages all states to be used at least once, but doesn't force equal usage.
Expected Outcome After Fixes
With these changes, the model should:
Temporal flickering in early epochs (healthy exploration)
Gradual stabilization by epoch 20-30 (not epoch 10)
3-6 active states (matching true behavior count)
Mean bout length: 20-100 frames (ethologically valid)
Temporal loss: 0.1-0.3 (some switching, not frozen)
Reconstruction MSE: 0.5-0.8 (slightly worse than collapsed model, but meaningful)
Implementation Priority
Must fix (run dies without these):
Fix #2 (loss weights) - 1 line change
Fix #4 (reduce states) - 1 line change
Should fix (improves quality): 3. Fix #1 (temporal convolution) - architectural change 4. Fix #3 (annealing) - requires passing epoch to forward() Nice to have: 5. Fix #5 (usage regularization) - alternative approach
Summary
Diagnosis: Severe codebook collapse + degenerate segmentation due to encoder lacking temporal context and codebook loss forcing uniform usage. Decision: REJECT - Model is scientifically useless in current state. Action: Implement Fixes #1, #2, #4 minimally. Retrain for 100 epochs. Expected training time: ~30-45 minutes on RTX 2080 Ti.