Scientific Diagnosis & Decision
1. The Diagnosis
Critical Failure Analysis: STILL COLLAPSED
Despite implementing all recommended fixes, the model has failed again with the exact same failure mode: Evidence of Persistent Collapse:
Codebook Loss Trajectory:
Epoch 1:  0.46  (healthy)
Epoch 10: 1.21  (collapsing)
Epoch 20: 1.79  (collapsed)
Epoch 100: 1.79  (still collapsed - max entropy for 6 states = log(6) = 1.79)
Temporal Change Rate:
Epoch 1-4: ~0.5-0.6 (healthy flickering)
Epoch 10: 0.23 (collapsing)
Epoch 20+: 0.0000 (completely frozen - NO state transitions)
The Death Spiral Repeated:
Codebook loss reached exactly 1.79 (≈ log(6)) → using 1 state out of 6
Temporal loss: 0.0 → states never change
All warnings: DEGENERATE_SEGMENTATION from epoch 10-100
Why The Fixes Didn't Work
The problem is deeper than I diagnosed:
Temporal convolution added ✓ But...
The GraphEncoder output still has no incentive to vary over time
The switching policy sees identical latent codes for all frames
Loss weights changed ✓ But...
gamma=0.0 removed the penalty, but there's still no reward for using multiple states
The model found it's still easier to collapse than to discover structure
Temperature annealing ✓ But...
By epoch 10, temperature was ~0.8, states had already collapsed
Annealing schedule was too fast (should be 100 epochs, not 50)
Reduced states to 6 ✓ But...
Still collapsed to 1 state - the problem isn't the number, it's the lack of incentive to switch
Root Cause: The Decoder Has No State-Conditional Information
Looking at the code flow:
# In forward():
latent = self.encode_to_latent(x)           # (B, T, 32)
state_logits = self.switching_policy(latent) # Uses latent
states = argmax(state_logits)                # (B, T)
reconstructed = self.decoder(latent, states) # ← Problem!
The decoder receives BOTH latent and states, but the latent is the same regardless of state. The state-dependent linear dynamics (A matrices) have no effect because:
Latent is computed BEFORE state assignment
Decoder uses latent directly, not "future latent after dynamics"
Model learns: "I can reconstruct perfectly with just the latent, ignoring states"
Result: States become irrelevant → collapse to single state.
2. The Decision: OPTION A - REFINE (Fundamental Architecture Fix Required)
Why This Model Is STILL Not Scientifically Useful
Same as before:
Zero segmentation: 1 state out of 6
No dynamics: Temporal change rate = 0.0
Invalid generative model: Cannot generate (ERROR during evaluation)
Trivial solution: Model is just an autoencoder with a useless discrete bottleneck
3. CRITICAL FIX REQUIRED: Redesign Decoder to Enforce State Usage
The current architecture allows the model to bypass the discrete bottleneck. We need to force the decoder to use state information.
Fix #1: Remove Direct Latent Path from Decoder (CRITICAL)
The decoder should ONLY see the state, not the continuous latent. File: model.py Current (broken) architecture:
def forward(self, x):
    latent = self.encode_to_latent(x)
    state_logits = self.switching_policy(latent)
    states = argmax(state_logits)
    reconstructed = self.decoder(latent, states)  # ← Uses both!
    return reconstructed, states
Fixed architecture:
def forward(self, x):
    latent = self.encode_to_latent(x)
    state_logits = self.switching_policy(latent)
    states = argmax(state_logits)
    
    # CRITICAL: Decoder sees ONLY the state, not the latent
    # Force model to encode all information into discrete states
    reconstructed = self.decoder_from_states(states)  # ← Only states!
    
    return reconstructed, states
Implementation: Replace the LinearDynamicsDecoder with a pure discrete decoder:
class DiscreteStateDecoder(nn.Module):
    """
    Decoder that ONLY uses discrete states (no continuous bypass)
    Forces model to encode all information into state sequence
    """
    def __init__(self, n_states, output_dim, hidden_dim=256):
        super().__init__()
        self.n_states = n_states
        self.output_dim = output_dim
        
        # State embedding (each state has its own learned embedding)
        self.state_embed = nn.Embedding(n_states, hidden_dim)
        
        # Temporal decoder (autoregressive over states)
        self.temporal_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, states):
        """
        Input: states (Batch, Time) - discrete state indices
        Output: (Batch, Time, output_dim) - reconstructed features
        """
        batch_size, seq_len = states.shape
        
        # Embed states
        state_emb = self.state_embed(states)  # (B, T, hidden_dim)
        
        # Apply temporal dynamics (GRU captures state transitions)
        temporal_features, _ = self.temporal_decoder(state_emb)  # (B, T, hidden_dim)
        
        # Project to output
        output = self.output_proj(temporal_features)  # (B, T, output_dim)
        
        return output
In DiscoveryPipeline.init:
# OLD:
self.decoder = LinearDynamicsDecoder(n_states, latent_dim, output_dim)

# NEW:
self.decoder = DiscreteStateDecoder(n_states, output_dim, hidden_dim=128)
In DiscoveryPipeline.forward():
# OLD:
reconstructed = self.decoder(latent, states_hard)

# NEW:
reconstructed = self.decoder(states_hard)  # Only states, no latent!
In DiscoveryPipeline.decode():
# Already correct - only uses states
reconstructed = self.decoder(codes)
Fix #2: Slow Down Temperature Annealing (CRITICAL)
Current annealing is too fast (50 epochs). Model collapses before exploration completes. File: model.py, line 412 Current:
epoch_fraction = min(self.current_epoch / 50, 1.0)  # Too fast!
Fixed:
epoch_fraction = min(self.current_epoch / 80, 1.0)  # Slower: anneal over 80 epochs
Fix #3: Add Entropy Bonus (Not Penalty)
Current gamma=0.0 removes pressure, but we need positive reward for using multiple states. File: loss.py, codebook loss section Current:
if gamma == 0.0:
    codebook_loss = 0.0  # No pressure at all
Fixed - Add NEGATIVE entropy (reward diversity):
# Instead of penalizing low entropy, REWARD high entropy
# Flip the sign: we WANT to maximize entropy, so minimize negative entropy
code_counts = torch.bincount(codes.flatten(), minlength=n_codes)
code_probs = code_counts.float() / codes.flatten().numel()
code_probs = code_probs + 1e-10

# Entropy (higher is better - more states used)
entropy = -torch.sum(code_probs * torch.log(code_probs))

# Reward entropy by minimizing negative entropy
# But don't force uniform - just encourage > 1 state
min_acceptable_entropy = np.log(2)  # At least 2 states
codebook_loss = max(0, min_acceptable_entropy - entropy)  # Penalty if < 2 states
Set gamma=0.1 again with this new formulation.
Fix #4: Add State Diversity Loss
Explicitly penalize using only one state. File: loss.py, after temporal loss Add new component:
# 5. State diversity (penalize collapse to single state)
unique_states_per_sequence = []
for seq in range(codes.shape[0]):
    unique_states_per_sequence.append(len(torch.unique(codes[seq])))
mean_unique_states = torch.tensor(unique_states_per_sequence, device=codes.device).float().mean()

# Penalty if using < 2 states per sequence
diversity_loss = max(0, 2.0 - mean_unique_states)  # Force at least 2 states/sequence
Update total loss:
total_loss = (
    alpha * recon_loss +
    beta * commit_loss +
    gamma * codebook_loss +
    delta * temporal_loss +
    0.5 * diversity_loss  # NEW: Add diversity term
)
Expected Outcome After These Fixes
The key insight: By removing the continuous latent bypass, we force the model to use the discrete states. The reconstruction loss will initially be HIGHER (that's good - it means states matter!), then gradually improve as the model learns meaningful states. Expected Training:
Epochs 1-30: Higher reconstruction loss (~1.5-2.0), exploring states
Epochs 30-60: States stabilize, reconstruction improves
Epochs 60-100: Refinement, reconstruction ~0.8-1.2 (worse than collapsed model, but meaningful)
Target Metrics:
Codebook usage: 50-100% (3-6 states)
Temporal change rate: 0.05-0.2 (some switching, not frozen)
Mean bout length: 30-150 frames
Codebook loss: 0.5-1.2 (NOT 1.79!)
Implementation Priority
Must fix (model is fundamentally broken without this):
Fix #1: Remove latent bypass from decoder - ARCHITECTURAL CHANGE
Should fix: 2. Fix #2: Slow annealing (1 line) 3. Fix #3: Entropy bonus (modify loss.py) 4. Fix #4: Diversity loss (add to loss.py)
Summary
Diagnosis: Model architecture allows bypassing the discrete bottleneck via continuous latent path. Previous fixes addressed symptoms, not root cause. Decision: REJECT - Still scientifically useless. Root cause: Decoder uses both latent AND states, so it ignores states and collapses. Critical fix: Replace LinearDynamicsDecoder(latent, states) with DiscreteStateDecoder(states) - force information bottleneck through discrete states only. This is a fundamental architectural redesign, not just hyperparameter tuning. The current HSLDS architecture is incompatible with discrete behavior discovery as implemented.