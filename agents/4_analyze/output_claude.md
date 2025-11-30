Post-Mortem Analysis: HSLDS Behavior Discovery
1. The Crash: Technical Bug
Location: The crash occurred during evaluation when calling generate() or similar inference code. Root Cause: Interface mismatch between training and inference. The new DiscreteStateDecoder.forward() accepts only states as input:
def forward(self, states):  # Takes 1 argument + self
But somewhere in the inference pipeline (likely in generate() or decode()), code is still calling it with TWO arguments:
decoder(latent, states)  # Old interface - WRONG
This is a trivial bug—the decoder interface was changed during the architectural fix but not all call sites were updated. The test file caught it, but training proceeded anyway because the training loop uses the correct interface. Fix: Find all decoder calls outside the training loop and update them to decoder(states) only.
2. The "Freezing" Failure: Why Zero State Transitions?
The model completely froze by epoch 24 (Temporal Change = 0.0000) and never recovered. This is worse than flickering—the model learned to never switch states.
The Smoking Gun
Look at the loss trajectory:
Epochs 1-9: Temporal loss decreases rapidly (0.614 → 0.328)
Epochs 10-24: Model transitions toward single-state collapse (temporal → 0.0000)
Epochs 24-100: Frozen solid (temporal = 0.0000, codebook = 1.7918 ≈ log(6))
The codebook loss plateaus at exactly log(6), which is the entropy of a uniform prior with 6 states. This means the model learned zero information about which states to use—it's just randomly initialized embeddings that never get updated.
Why Did This Happen?
The temporal coherence penalty (delta=1.0) created a devastating feedback loop:
Epoch 1-5: Model starts exploring (temporal change ~50%)
Epoch 5-10: Temporal penalty kicks in, model reduces transitions
Epoch 10-15: Model discovers it can make temporal loss = 0 by never switching
Epoch 15+: Single-state attractor basin—impossible to escape
Once the model commits to one state for all data:
Temporal loss = 0.0 (perfect score!)
Reconstruction loss plateaus (~0.9, acceptable)
Codebook loss = log(6) (doesn't matter, gamma=0.0)
Entropy bonus (epsilon=0.01) is too weak to fight back
Diversity loss (zeta=0.5) on embeddings doesn't help if embeddings are never used
The model found a degenerate local minimum where using one state forever minimizes the loss better than using multiple states.
3. Root Cause: Why HSLDS Failed vs. Standard VQ-VAE
The Fundamental Problem: Switching Dynamics Without Dynamics
The HSLDS architecture assumes:
Each discrete state has associated linear dynamics (A, B matrices)
State transitions are meaningful because they change the dynamics
The decoder uses both the state and the latent to reconstruct
But we removed the linear dynamics system entirely! The "DiscreteStateDecoder" is just:
state_embedding → GRU → MLP → reconstruction
This is not a switching linear dynamical system—it's a discrete state VAE with temporal smoothing. The GRU provides temporal context, but there's no actual "dynamics" (no state-dependent transition matrices, no linear evolution equations).
Why VQ-VAE Would Work Better
A standard VQ-VAE would succeed here because:
No temporal penalty: Allows frame-by-frame independence
Simpler bottleneck: Just quantize latents, no switching dynamics
Proven architecture: Codebook learning is well-studied (EMA updates, commitment loss)
The HSLDS adds massive complexity (switching dynamics, temporal coherence, state-dependent models) but we:
Removed the actual dynamics (no A/B matrices)
Removed the latent bypass (forcing 100% through discrete states)
Added aggressive temporal penalty (forcing states to freeze)
We created a crippled VQ-VAE that's worse than both:
Worse than HSLDS: No actual dynamics, just discrete states
Worse than VQ-VAE: Aggressive temporal penalty prevents proper quantization
The Loss Function Contradiction
The loss weights created an impossible optimization:
alpha=1.0: Reconstruct well (needs expressive states)
delta=1.0: Don't change states (encourages single state)
epsilon=0.01: Use diverse states (too weak)
zeta=0.5: Make states different (doesn't matter if unused)
Result: The model chooses the path of least resistance—use one state forever, achieve temporal=0, accept mediocre reconstruction (~0.9 MSE).
4. The "Glass Ceiling": What Would Break Through?
If we had infinite time, here's the next major architectural change needed:
Option A: Explicit Transition Modeling (True HSLDS)
Add a learned transition model that makes switching states costly but meaningful:
class TransitionModel(nn.Module):
    def __init__(self, n_states):
        self.transition_probs = nn.Parameter(torch.randn(n_states, n_states))
    
    def forward(self, prev_state, curr_state):
        # Log probability of transitioning from prev_state to curr_state
        return F.log_softmax(self.transition_probs, dim=1)[prev_state, curr_state]
Add transition likelihood to loss:
transition_loss = -model.transition_model(states[:, :-1], states[:, 1:]).mean()
This creates structured transitions (some transitions are easy, others hard) rather than a uniform temporal penalty. The model learns when to switch, not just to minimize switching.
Option B: Hierarchical Temporal Abstraction
Replace the single discrete state with two-level hierarchy:
Slow states (behaviors): Change every ~20 frames
Fast states (sub-behaviors): Change every ~5 frames within a behavior
slow_states = gumbel_softmax(encoder(data), tau=low_temp)  # Sticky
fast_states = gumbel_softmax(encoder(data, slow_states), tau=high_temp)  # Flexible
Different temporal penalties for each level:
delta_slow = 2.0: Strongly penalize behavior switches
delta_fast = 0.1: Allow sub-behavior switches
This matches behavioral hierarchies in real animals (groom-still-walk vs. left-paw-right-paw-scratch).
Option C: Supervised Behavioral Priors (The Nuclear Option)
Use ground truth labels for the first 20 epochs:
if epoch < 20:
    supervised_loss = F.cross_entropy(logits, true_labels)
    total_loss += 10.0 * supervised_loss
This bootstraps the discrete states to match real behaviors, then fine-tunes unsupervised. It's "cheating" for discovery, but would prove whether the architecture can work at all.
The Brutal Truth
The HSLDS failed because:
Architecture mismatch: Built switching dynamics, removed the dynamics
Loss contradictions: Temporal penalty fought diversity objectives
Insufficient gradient signal: Discrete bottleneck + weak diversity loss = no learning
Degenerate attractor: Single-state solution is a strong local minimum
The model didn't fail to train—it successfully optimized the loss we gave it. We asked for minimal temporal changes, and it delivered: zero changes forever. To fix this properly, we need to either:
Go back to VQ-VAE: Remove temporal penalty, use proven codebook learning
Go forward to true HSLDS: Add transition models, linear dynamics, hierarchical states
Give up on pure discovery: Bootstrap with weak supervision
The middle ground—discrete states + aggressive temporal smoothing—is a death trap.