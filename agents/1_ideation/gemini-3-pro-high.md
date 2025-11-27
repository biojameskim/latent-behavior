# Phase 0: First-Principles Architecture Derivation
## 1. Derive Architectural Hypotheses
### Hypothesis 1: The "Quantized Body-Graph" (GNN-VQ-VAE)

- Core Assumption: Behavior is a "language" of discrete, reusable motifs (syllables) enacted by a structured, articulated body (the fly). The continuous trajectory is a sequence of these discrete atomic actions.
- Encoder Strategy $q(z|x)$: Spatio-Temporal GNN.
- Spatial: A Graph Neural Network (GNN) explicitly models the fly's morphology (nodes = keypoints like wing_left, head; edges = physical connections). This handles the 48-feature high-dimensional space by injecting the known biological structure.
- Temporal: 1D Convolutions (TCN) with dilated receptive fields to aggregate frame-level pose data into short-term kinematic embeddings.
- The Bottleneck: Discrete (Vector Quantized).
- We use a codebook of $K$ latent vectors. This forces the model to cluster continuous movements into $K$ distinct "behavioral words" (e.g., "wing_extension", "turn_left"), satisfying the segmentation requirement directly.
- Dynamics Model: Autoregressive Transformer (GPT-style).
Learns the probability $p(z_{t+1} | z_{t}, z_{t-1}, ...)$. It treats behavior generation as next-token prediction, enabling the synthesis of complex, long-horizon sequences.
- Why this fits Drosophila data:
Structure: Flies are articulated systems; a GNN respects this better than a flat MLP.
Discreteness: Ethograms (dictionaries of behaviors) are the standard ground truth in ethology. VQ-VAE naturally discovers this ethogram.
Timescales: 30Hz data captures pose changes (wing extension) rather than high-freq wing beats. These pose changes are well-modeled as discrete state transitions.

### Hypothesis 2: The "Switching Control System" (Recurrent Switching LDS)

- Core Assumption: The fly acts as a dynamical system with multiple "regimes" or "policies" (e.g., a "flight" controller vs. a "grooming" controller). Behavior is the switching process between these linear dynamical laws.
- Encoder Strategy $q(z|x)$: Bi-directional LSTM.
Processes the full trajectory context to infer the current "regime" $z_t$ and the continuous state $h_t$.
- The Bottleneck: Factorized (Discrete $z_t$ + Continuous $h_t$).
$z_t$: The discrete "mode" (segmentation).
$h_t$: The continuous physical state (position, velocity).
- Dynamics Model: State-Dependent Transition.
$h_{t+1} = A_{z_t} h_t + b_{z_t} + \epsilon$. The transition matrix $A$ depends on the discrete mode $z_t$.
- Why this fits Drosophila data:
Physics: It explicitly models the Newtonian dynamics of movement, which is appropriate for trajectory data.
Segmentation: It defines behaviors by their dynamics (how they move) rather than just their appearance (how they look), which is crucial for distinguishing "running" from "walking".

### Hypothesis 3: The "Social-Field Manifold" (Contrastive ODE-Flow)

- Core Assumption: Behavior is a continuous flow on a low-dimensional manifold, shaped by an "energy landscape" defined by social potentials (e.g., attraction to the other fly).
- Encoder Strategy $q(z|x)$: Social-Attention Encoder.
- Uses Self-Attention to weigh the importance of different body parts and the social feature (min_fg_dist) dynamically.
- The Bottleneck: Continuous & Smooth (Gaussian Process or VAE).
- Regularized to be smooth over time, avoiding "jumpy" latent spaces.
- Dynamics Model: Neural ODE (Ordinary Differential Equation).
- Models $\frac{dz}{dt} = f(z, \text{social_context})$. This allows for continuous-time generation and handles irregular sampling if necessary.
- Why this fits Drosophila data:
Sociality: Explicitly models the interaction as a force field.
Fluidity: Produces the most realistic, smooth trajectories for synthesis, avoiding the "jerky" artifacts of discrete models.

## 2. Evaluation Criteria Matrix
Hypothesis	Interpretability	Temporal Modeling	Robustness to Noise	Generative Quality	Complexity	Overall Score
Hyp 1: Quantized Body-Graph	5/5 (Discrete codes = Ethogram)	5/5 (Transformer handles long-range)	4/5 (VQ filters noise)	4/5 (Good, but discrete jumps possible)	4/5 (High, but modular)	4.6
Hyp 2: Switching LDS	4/5 (Regimes are clear)	3/5 (Markovian/RNN limits)	3/5 (Sensitive to transition noise)	3/5 (Linear dynamics can be rigid)	5/5 (Very complex inference)	3.8
Hyp 3: Social Manifold	2/5 (Latents are abstract)	4/5 (ODE is powerful)	5/5 (Smoothness prior removes noise)	5/5 (Fluid, realistic motion)	3/5 (ODEs are slow to train)	3.8

## 3. Selection Decision
Selected Architecture: Hypothesis 1 (The "Quantized Body-Graph")

Justification:

Mathematically Optimal for 30Hz Interaction:

- The 30Hz sampling rate is too slow for fine-grained physics (making ODEs/LDS less advantageous) but perfect for capturing "pose sequences".
- The GNN encoder correctly handles the non-Euclidean geometry of the 24 keypoints (wings are far from legs in the array, but connected in the body).
- The VQ-VAE structure solves the "Zero Ground Truth" problem by forcing the model to discover a vocabulary of $K$ behaviors to minimize reconstruction error.
Segmentation vs. Generation Balance:

- Segmentation: The Vector Quantization layer provides hard, interpretable segment boundaries (whenever the code index changes).
- Generation: The Transformer dynamics model (trained on the discrete codes) can learn complex, multi-modal transition probabilities (e.g., "after grooming, usually walk or rest"), which linear models (Hyp 2) struggle with.

Failure Modes & Detection:

- Codebook Collapse: The model might use only 1-2 codes. Detection: Monitor perplexity of the codebook usage.
- Flickering: Rapid switching between codes. Detection: Measure average segment duration; if < 3 frames, apply temporal smoothing or penalties.
- Loss of Social Context: The model might ignore the min_fg_dist feature. Detection: Correlate latent codes with min_fg_dist to ensure social states are captured.

Alternatives:

- Alternative 1: Hypothesis 2 (Switching LDS) if we find the VQ-VAE fails to capture the dynamics (velocity/acceleration) of the movements.
- Alternative 2: A hybrid "Hierarchical VQ-VAE" to capture multiple timescales if the single timescale proves insufficient.