# HSLDS Architecture Specification

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HSLDS DISCOVERY PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

INPUT: Raw Keypoints (Batch, Time, 48)
   │
   │  [48 = 24 keypoints × 2 coords (x,y)]
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BEHAVIOR PREPROCESSOR                          │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  Ego-Centric   │→ │   Velocities   │→ │    Features      │  │
│  │   Alignment    │  │  (Temporal ∆)  │  │  (Distances,     │  │
│  │  (Remove COM)  │  │                │  │   Angles, etc.)  │  │
│  └────────────────┘  └────────────────┘  └──────────────────┘  │
│                              ↓                                   │
│                      ┌────────────────┐                          │
│                      │  Normalization │                          │
│                      │   (Z-score)    │                          │
│                      └────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
   │
   │  PROCESSED: (Batch, Time, 121)
   │  [48 positions + 48 velocities + 25 engineered features]
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GRAPH ENCODER (GNN)                         │
│                                                                  │
│  For each timestep:                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  24 Keypoints (12 per fly)                               │   │
│  │                                                           │   │
│  │  Fly 1:                      Fly 2:                      │   │
│  │  ┌─────────────┐             ┌─────────────┐            │   │
│  │  │ GCN Layer 1 │             │ GCN Layer 1 │            │   │
│  │  │  (2D → 16D) │             │  (2D → 16D) │            │   │
│  │  └──────┬──────┘             └──────┬──────┘            │   │
│  │         │ Message Passing           │                    │   │
│  │  ┌──────▼──────┐             ┌──────▼──────┐            │   │
│  │  │ GCN Layer 2 │             │ GCN Layer 2 │            │   │
│  │  │ (16D → 32D) │             │ (16D → 32D) │            │   │
│  │  └──────┬──────┘             └──────┬──────┘            │   │
│  │         │                           │                    │   │
│  │  ┌──────▼──────┐             ┌──────▼──────┐            │   │
│  │  │ GCN Layer 3 │             │ GCN Layer 3 │            │   │
│  │  │(32D → 32D)  │             │(32D → 32D)  │            │   │
│  │  └──────┬──────┘             └──────┬──────┘            │   │
│  │         │                           │                    │   │
│  │  ┌──────▼───────────────────────────▼──────┐            │   │
│  │  │    Global Mean Pool (per fly)           │            │   │
│  │  └──────┬───────────────────────┬──────────┘            │   │
│  │         │                       │                        │   │
│  │  ┌──────▼───────────────────────▼──────┐                │   │
│  │  │     Aggregation (Sum/Concat)        │                │   │
│  │  │          → 32D latent               │                │   │
│  │  └─────────────────────────────────────┘                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
   │
   │  LATENT: (Batch, Time, 32)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SWITCHING POLICY (RNN)                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  GRU(input_size=32, hidden_size=64)                    │     │
│  │    ↓                                                    │     │
│  │  Hidden Context: (Batch, Time, 64)                     │     │
│  └────────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Transition Network:                                   │     │
│  │    Input: [hidden_64D, prev_state_one_hot_12D]         │     │
│  │    ↓                                                    │     │
│  │    Linear(76 → 128) + ReLU                             │     │
│  │    ↓                                                    │     │
│  │    Linear(128 → 12) [n_states logits]                  │     │
│  │    ↓                                                    │     │
│  │    + Sticky Bias (encourage self-transitions)          │     │
│  └────────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Gumbel-Softmax Sampling (differentiable)              │     │
│  │    → Hard state selection                              │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
   │
   │  STATES: (Batch, Time) ∈ {0, 1, ..., 11}
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│              LINEAR DYNAMICS DECODER (State-Dependent)           │
│                                                                  │
│  State-Specific Parameters:                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  A_matrices: [n_states, latent_dim, latent_dim]       │     │
│  │             = [12, 32, 32]                             │     │
│  │                                                         │     │
│  │  b_vectors:  [n_states, latent_dim]                    │     │
│  │             = [12, 32]                                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│  For each (latent_t, state_t):                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  1. Select A_{state_t}, b_{state_t}                    │     │
│  │  2. Apply dynamics: z'_t = A @ z_t + b                 │     │
│  │  3. Emit observation:                                   │     │
│  │     Linear(32 → 128) + ReLU                            │     │
│  │     Linear(128 → 256) + ReLU                           │     │
│  │     Linear(256 → 121)                                  │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
   │
   │  RECONSTRUCTED: (Batch, Time, 121)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOSS COMPUTATION                            │
│                                                                  │
│  L_total = α·L_recon + β·L_commit + γ·L_codebook + δ·L_temporal │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ L_recon = MSE(original, reconstructed)              │       │
│  │         [How well can we reconstruct?]               │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ L_commit = CrossEntropy(state_logits, argmax_states) │       │
│  │          [Are codes stable?]                         │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ L_codebook = log(K) - H(p(states))                  │       │
│  │            [Are all K states used?]                  │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ L_temporal = E[s_t ≠ s_{t-1}]                       │       │
│  │            [Do states persist over time?]            │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: Discrete behavioral codes + Reconstructed trajectories
```

## Mathematical Formulation

### 1. Preprocessing

```
x_raw ∈ ℝ^(B×T×48)   [Raw keypoints]

x_ego = EgoCenter(x_raw)           [Remove global position]
x_vel = ∇_t x_ego                  [Temporal derivatives]
x_feat = Engineer(x_ego)           [Distances, angles, etc.]
x_combined = [x_ego; x_vel; x_feat] ∈ ℝ^(B×T×121)
x = Normalize(x_combined)          [Z-score]
```

### 2. Encoding (Graph Neural Network)

For each timestep t:

```
Per-fly GNN:
  h^(0) = x_kp ∈ ℝ^(12×2)          [12 keypoints, 2D coords]
  h^(1) = ReLU(GCN₁(h^(0), E))     → ℝ^(12×16)
  h^(2) = ReLU(GCN₂(h^(1), E))     → ℝ^(12×32)
  h^(3) = GCN₃(h^(2), E)           → ℝ^(12×32)
  z_fly = MeanPool(h^(3))          → ℝ^32

Combine flies:
  z_t = f(z_fly1, z_fly2)          → ℝ^32
```

Where E is the edge connectivity (anatomical graph).

### 3. Switching Dynamics

```
Temporal context:
  c_t = GRU(z_t, c_{t-1})          ∈ ℝ^64

State transition logits:
  u_t = MLP([c_t; s_{t-1}])        ∈ ℝ^K
  π_t = u_t + sticky_bias · 𝟙[s_t = s_{t-1}]

Discrete state:
  s_t ~ Categorical(softmax(π_t))  ∈ {0, ..., K-1}
```

### 4. State-Dependent Linear Dynamics

```
Dynamics:
  z'_t = A_{s_t} z_t + b_{s_t} + ε  where ε ~ N(0, σ²I)

Emission:
  x̂_t = MLP(z'_t)                  ∈ ℝ^121
```

### 5. Loss Function

```
L = α · ||x - x̂||² +                        [Reconstruction]
    β · CrossEntropy(π, argmax(π)) +        [Commitment]
    γ · (log K - H(p(s))) +                 [Codebook usage]
    δ · 𝔼[s_t ≠ s_{t-1}]                    [Temporal coherence]
```

Default weights: α=1.0, β=0.25, γ=0.1, δ=0.5

## Parameter Counts

```
Component                Parameters
─────────────────────────────────────────
GraphEncoder
  GCNConv(2→16)          ~544
  GCNConv(16→32)         ~1,024
  GCNConv(32→32)         ~2,048
  Readout MLP            ~2,080
  Subtotal:              ~5,696

SwitchingPolicy
  GRU(32→64)             ~18,816
  Transition MLP         ~17,664
  Sticky bias            12
  Subtotal:              ~36,492

LinearDynamicsDecoder
  A_matrices (12×32×32)  12,288
  b_vectors (12×32)      384
  Emission MLP           ~48,000
  Subtotal:              ~60,672

─────────────────────────────────────────
TOTAL:                   ~102,860 parameters
```

## Inductive Biases

1. **Graph structure**: Body parts are connected anatomically
2. **Discrete states**: Behaviors are distinct, not continuously varying
3. **Linear dynamics**: Kinematics are approximately linear within syllables
4. **Temporal persistence**: Behaviors persist (sticky transitions)
5. **Ego-centric**: Position-invariant (relative coordinates)

## Design Rationale

| Choice | Justification |
|--------|---------------|
| **GNN encoder** | Keypoints have known anatomical connectivity; message passing captures biomechanical constraints |
| **Discrete bottleneck** | Enables crisp segmentation for neuroscience interpretation |
| **Linear dynamics per state** | 30Hz data exhibits ~linear kinematics over 100-300ms behavioral syllables |
| **Sticky prior** | Real behaviors persist (grooming lasts multiple frames) |
| **Multi-objective loss** | Balance reconstruction, codebook usage, and temporal coherence |

## Comparison to VQ-VAE

| Aspect | HSLDS | VQ-VAE |
|--------|-------|--------|
| **Encoder** | GNN (graph structure) | CNN/MLP (flat) |
| **Bottleneck** | Discrete states with dynamics | Discrete codes (no dynamics) |
| **Decoder** | State-dependent linear + MLP | MLP only |
| **Temporal model** | Recurrent switching policy | Separate prior (e.g., PixelCNN) |
| **Interpretability** | High (states = behaviors) | Medium (codes = poses) |
| **Generative** | Autoregressive states + dynamics | Requires separate prior model |

## Data Flow Example

```
Input sequence (1 fly pair, 300 frames):
  x_raw: (1, 300, 48)

Preprocessing:
  x: (1, 300, 121)

Encoding (per timestep):
  For each t ∈ [1, 300]:
    z_t: (1, 32)

Switching:
  GRU context: (1, 300, 64)
  States: (1, 300) ∈ {0..11}
    Example: [3, 3, 3, 7, 7, 7, 7, 1, 1, 1, ...]
             └─────┘ └───────┘ └─────┘
              Bout 1   Bout 2   Bout 3

Decoding:
  x̂: (1, 300, 121)

Loss:
  Recon: MSE(x, x̂)
  Temporal: #switches / 299 = 2/299 ≈ 0.0067 (good!)
  ...
```

## Extension Points

1. **Hierarchical**: Add meta-states over primitive states
2. **Variational**: Replace deterministic encoder with q(z|x)
3. **Attention**: Add fly1 ↔ fly2 cross-attention
4. **Non-linear dynamics**: Replace A_k with neural ODE
5. **Multi-modal**: Add audio, optogenetic stimulation, etc.

## References

**Switching Linear Dynamical Systems**:
- Ghahramani & Hinton (2000). Variational learning for switching state-space models
- Fox et al. (2011). A sticky HDP-HMM with application to speaker diarization

**Graph Neural Networks**:
- Kipf & Welling (2017). Semi-supervised classification with graph convolutional networks

**Behavior Modeling**:
- Wiltschko et al. (2015). Mapping sub-second structure in mouse behavior
- Markowitz et al. (2018). The striatum organizes 3D behavior via moment-to-moment action selection

**VQ-VAE Baseline**:
- van den Oord et al. (2017). Neural discrete representation learning
