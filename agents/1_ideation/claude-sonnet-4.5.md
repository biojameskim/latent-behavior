# Phase 0: First-Principles Architecture Derivation

## 1. Derived Architectural Hypotheses

---

## Hypothesis 1: Hierarchical Switching Linear Dynamical System (HSLDS)

**Core Assumption:** Behavior is composed of discrete primitive motor programs, each governed by linear dynamics in a low-dimensional latent space. Social interactions emerge from coordinated switching between primitives across both agents.

### Encoder Strategy q(z|x):
Decompose 48D → Two 24D agents (leveraging domain knowledge that 24 keypoints = 1 fly)

**Graph Neural Network for each agent:**
- Nodes = 24 keypoints  
- Edges = anatomical connectivity (head-thorax-abdomen chain, leg pairs, wings)  
- Message passing captures biomechanical constraints  

**Cross-agent attention layer:** Captures social context (e.g., relative positions)

**Output:** 16D continuous latent per agent → 32D combined latent

### The Bottleneck:
- Discrete switching variable (K states, e.g., K=8–20)  
- Continuous dynamics within each state  
- Gaussian emissions from latent → observations

**Why:** Drosophila exhibit clear behavioral syllables (grooming stroke, wing flick) but smooth execution within syllables.

### Dynamics Model:
z_t = A_{s_t} z_{t-1} + b_{s_t} + ε
s_t ~ Categorical(π(s_{t-1}, context))

markdown
Copy code
- State-dependent linear dynamics (A matrix switches)  
- Recurrent switching policy conditioned on history  

### Why this fits Drosophila data:
- **Ethological validity:** Fruit flies perform stereotyped action sequences (grooming, courtship displays)  
- **Linear dynamics hypothesis:** Within a motor primitive, kinematics are approximately linear over short timescales (50–100ms)  
- **Interpretability:** Discrete states map to nameable behaviors; A matrices reveal kinematic structure  
- **30Hz suitability:** Linear systems naturally handle fixed sample rates  

---

## Hypothesis 2: Temporal Convolutional Variational Autoencoder (TC-VAE)

**Core Assumption:** Behavior is a continuous flow on a nonlinear manifold, not discrete switches. The key is learning multi-scale temporal receptive fields to capture both fast reflexes and slow postural adjustments.

### Encoder Strategy q(z|x):
**1D Temporal Convolutions over the 300-frame sequence**

**Multi-scale architecture:**
- **Fast pathway:** kernel=3, dilation=1,2,4 (captures 10–40ms dynamics)  
- **Slow pathway:** kernel=9, dilation=1,4,16 (captures 0.3–1.6s dynamics)  

No explicit agent decomposition: Let convolutions discover structure.  
Positional embeddings: Sine/cosine encoding to preserve temporal order.

**Output:** 128D latent code per timestep

### The Bottleneck:
- Continuous, dense latent space (VQ-VAE style but without quantization)  
- Sparse regularization (L1 on latent activations)

**Why:** Avoids hard segmentation; allows gradual transitions and co-occurring behaviors.

### Dynamics Model:
- Autoregressive decoder: Causal 1D convolutions  
- Predict x_{t+1:t+k} from z_{1:t}  
- Trained with teacher forcing + scheduled sampling  

### Why this fits Drosophila data:
- No assumptions about syllable boundaries  
- Multi-scale: Captures wing vibrations + walking  
- Translation invariance  
- Smooth interpolation → novel behavior blends  

---

## Hypothesis 3: Graph Temporal Transformer with VQ-Codebook (GTT-VQ)

**Core Assumption:** Behavior is a compositional language of body part configurations. Like words in a sentence, there exists a finite vocabulary of pose atoms that compose via temporal syntax.

### Encoder Strategy q(z|x):

**Per-timestep processing:**
- Spatial Graph Transformer  
- Learnable edge weights  
- Captures pose geometry  

**Temporal Transformer:**
- Self-attention over time  
- Positional encoding  
- Captures sequential dependencies  

**Hierarchical:**
- Local temporal windows (30 frames) → pose codes  
- Global sequence → behavior codes  

### The Bottleneck:
**Vector-Quantized (VQ) Codebook:**
- Spatial codebook: 64 pose prototypes  
- Temporal codebook: 32 behavior motifs  

Two-level discretization:
- Frame-level: Which pose?  
- Sequence-level: Which behavior sequence?  

**Why:** Interpretable vocabulary; compositional generation.

### Dynamics Model:
- Transformer decoder (GPT-style)  
- Autoregressively predicts next pose code  
- Trained with cross-entropy  

### Why this fits Drosophila data:
- Compositional behavior  
- Long-range dependencies (courtship)  
- Interpretable pose dictionary  
- Generative sampling from transitions  

---

## Hypothesis 4: Recurrent State Space Model with Normalizing Flows (RSSM-Flow)

**Core Assumption:** Behavior has stochastic, non-Markovian dynamics with both deterministic memory and stochastic transitions.

### Encoder Strategy q(z|x):
- Bidirectional GRU  
- Per-agent processing (24D → GRU → 16D)  
- Fusion via MLP → 32D combined latent  

### The Bottleneck:
- Stochastic latent  
- Deterministic state  
- Normalizing flow for non-Gaussian posterior  

### Dynamics Model:
h_t = GRU(h_{t-1}, z_{t-1})
z_t ~ Flow(h_t, x_t)
x̂_t ~ p(x | h_t, z_t)

yaml
Copy code

### Why this fits Drosophila data:
- Handles noise, occlusions  
- Captures multimodal choices  
- RNN memory for long dependencies  
- Diverse generative futures  

---

## Hypothesis 5: Disentangled β-TCVAE with Kinematic Priors

**Core Assumption:** Observations arise from factorized latent factors (position, velocity, angle, wing state, social distance).

### Encoder Strategy q(z|x):
Factorized architecture:
- Position encoder  
- Velocity encoder  
- Angular encoder  
- Social encoder  

TC penalty enforces independence.

**Total latent:**  
z = [z_pos, z_vel, z_angle, z_social] = 18D

### The Bottleneck:
Continuous, factorized latent structure.

### Dynamics Model:
**Kinematic Kalman Filter:**
- Position ← velocity  
- Velocity with inertia  
- Angles follow rotational dynamics  
- Learned transitions for other factors  

### Why fits Drosophila data:
- Interpretable factors  
- Physics-informed  
- Factor manipulation  
- Robust to missing data  

---

# 2. Evaluation Criteria Matrix

| Hypothesis | Interpretability | Temporal Modeling | Robustness to Noise | Generative Quality | Complexity | Overall Score |
|-----------|------------------|-------------------|----------------------|--------------------|------------|---------------|
| HSLDS     | 5 | 4 | 3 | 4 | 3 | 19/25 |
| TC-VAE    | 2 | 5 | 4 | 5 | 2 | 18/25 |
| GTT-VQ    | 5 | 5 | 3 | 4 | 1 | 18/25 |
| RSSM-Flow | 3 | 5 | 5 | 5 | 2 | 20/25 |
| β-TCVAE   | 5 | 3 | 4 | 3 | 3 | 18/25 |

### Justifications:
- **Interpretability:** HSLDS, GTT-VQ, β-TCVAE strongest  
- **Temporal Modeling:** RSSM-Flow, GTT-VQ, TC-VAE best  
- **Robustness:** RSSM-Flow best  
- **Generative Quality:** RSSM-Flow, TC-VAE strongest  
- **Complexity:** TC-VAE simplest; GTT-VQ most complex  

---

# 3. Selection Decision

## PRIMARY SELECTION: Hypothesis 1 (HSLDS) with modifications

### Justification:

#### Mathematical Optimality for 30Hz:
- High temporal correlation at 30Hz  
- Linear dynamics valid within syllables  
- Switching handles non-stationarity cleanly  
- Discrete bottleneck gives segmentation  

#### Segmentation ↔ Generation Tradeoff:
- Discrete states → crisp segmentation  
- Continuous within-state dynamics → smooth generation  
- Sticky transitions → behavioral persistence  

### Modifications:
- Graph NN encoder for anatomical structure  
- Social context vector in switching  
- Sticky HMM prior  

---

## Anticipated Failure Modes & Detection

| Failure Mode | Symptom | Detection Method |
|--------------|---------|------------------|
| State collapse | Uses only 2–3 states | Monitor state entropy |
| Overfitting | Perfect train recon, poor gen | Holdout likelihood |
| Linear limits | High within-state error | Per-state residuals |
| Misaligned segmentation | Behavior mismatch | Compare to labels |

---

## Why not the others?

- **RSSM-Flow:** Second best; lacks discrete segmentation  
- **TC-VAE:** Great at generation, bad at segmentation  
- **GTT-VQ:** Too complex for 50 sequences  
- **β-TCVAE:** Factorized but linear dynamics may fail for interactions  

---

## Alternative 1: RSSM-Flow
If HSLDS states are unstable/uninterpretable:
- Smooth continuous latent  
- Post-hoc clustering  
- Better noise handling  

## Alternative 2: GTT-VQ
If behaviors are highly compositional:
- Interpretable pose vocabulary  
- Sparse pose activation  
- Discovery of reusable pose atoms  

---

**STOP. Awaiting approval to proceed with HSLDS implementation or pivot to an alternative.**