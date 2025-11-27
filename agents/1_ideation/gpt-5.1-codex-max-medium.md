Phase 0: First-Principles Architecture Derivation

### Hypothesis 1: Switching Graph VAE-HMM

- Core Assumption: Behavior is a discrete set of latent motifs that switch Markovianly; each motif generates smooth pose dynamics.
- Encoder q(z|x): Framewise graph encoder over 24 keypoints (undirected k-NN on fly skeleton) using GCN layers + temporal 1D conv to aggregate short context; outputs categorical logits.
- Bottleneck: Discrete (Gumbel-softmax) state with moderate cardinality (e.g., 15–30). Encourages interpretable motifs and segmentation.
- Dynamics Model: HMM transition over discrete states; state-conditioned linear-Gaussian (or small MLP) for Δpose to predict t+1; optional duration prior to reduce rapid flicker.
- Why fits Drosophila data: Distinct actions (groom, wing-extension) are quasi-stationary and recur; discreteness matches ethogram-like interpretation; transitions capture bout structure at 30 Hz.

### Hypothesis 2: Hierarchical Semi-Markov VAE (HSMM-VAE)

- Core Assumption: Behaviors are hierarchical: coarse bouts made of micro-movements; durations are non-geometric.
- Encoder q(z|x): Two-level encoder: (i) frame-level temporal CNN + keypoint graph to produce micro-state embeddings; (ii) chunk-level bi-GRU over downsampled embeddings to infer segment-level discrete codes.
- Bottleneck: Hierarchical discrete codes: slow-timescale segment code (semi-Markov duration) and fast-timescale micro-state (categorical). Captures multi-timescale structure.
- Dynamics Model: HSMM for segment codes with learned duration distributions; within each segment, AR(1) Gaussian with code-conditioned parameters for Δpose; optional autoregressive normalizing flow for richer within-segment dynamics.
- Why fits Drosophila data: Grooming/walking bouts have characteristic lengths; hierarchical model separates bout identity from within-bout kinematics; reduces over-segmentation at 30 Hz.

### Hypothesis 3: Continuous Manifold Flow with Event Sparsity (Flow-VAE + Sparse Events)

- Core Assumption: Behavior evolves on a low-dimensional continuous manifold with occasional sparse event-driven deviations (e.g., wing flicks).
- Encoder q(z|x): Temporal transformer with keypoint-wise attention (treat keypoints as tokens) to capture cross-part coupling; outputs Gaussian latent and Bernoulli event logits.
- Bottleneck: Continuous latent (d=8–12) for smooth dynamics + sparse binary event channel (relaxed Bernoulli) to mark abrupt deviations.
- Dynamics Model: Latent neural ODE or GRU to evolve continuous z; event channel gates an additive residual (small MLP) for Δz predicting t+1; decoder is flow-based to map z→pose for sharp details.
- Why fits Drosophila data: Captures smooth locomotion manifold while allowing sharp, sparse actions (wing extension, rapid turns); attention handles non-local keypoint coupling (wings–legs–body).

### Hypothesis 4: Neural AR-HSMM with Pose-Diffusion Decoder

- Core Assumption: Discrete semi-Markov states govern coarse behavior; pose within a state follows a diffusion prior for high-fidelity realism.
- Encoder q(z|x): Temporal CNN + keypoint graph; infers segment-level discrete state with duration; also produces conditioning embeddings for diffusion decoder.
- Bottleneck: Discrete semi-Markov states; continuous conditioning for diffusion steps.
- Dynamics Model: Semi-Markov transitions; within a segment, AR(1) latent; decoder is conditional diffusion over poses (or residual Δpose) conditioned on state and past few frames.
- Why fits Drosophila data: Semi-Markov avoids state-chatter; diffusion decoder improves generative sharpness of fine wing/leg configurations.

### Hypothesis 5: Factorized Multi-Agent Interaction Model (Bi-Fly Graph VAE)

- Core Assumption: Social interaction = two coupled agents with partly independent behaviors and an interaction latent.
- Encoder q(z|x): Split keypoints into fly A/B; per-fly GCN + temporal CNN encoders → per-fly latent; cross-attention to infer interaction latent capturing relative pose.
- Bottleneck: Factorized latent: discrete per-fly state + continuous interaction latent (relative geometry).
- Dynamics Model: Coupled HMMs (per-fly) with interaction-dependent transition biases; relative-geometry latent evolves via small GRU; decoder combines per-fly pose decoders conditioned on interaction latent to predict t+1.
- Why fits Drosophila data: Many behaviors are contingent (chase, courtship); factorization improves interpretability per fly and captures dyadic coupling.

## Evaluation Criteria Matrix (1–5)

|Hypothesis|Interpretability|Temporal Modeling|Robustness to Noise|Generative Quality|Complexity|Overall Score|
|---|---|---|---|---|---|---|
|Hyp 1|4 (clear discrete states)|3 (Markov + AR)|4 (discrete denoises)|3 (simple decoder)|2 (moderate)|3.2|
|Hyp 2|4 (hierarchical bouts)|5 (semi-Markov + hierarchy)|4 (durations reduce chatter)|4 (richer AR/flow)|3 (higher)|4.0|
|Hyp 3|3 (continuous less direct)|4 (ODE/GRU + events)|3 (continuous sensitive)|4 (flow decoder)|3 (higher)|3.4|
|Hyp 4|3 (states + diffusion)|5 (semi-Markov)|4 (state smoothing)|5 (diffusion fidelity)|4 (high)|4.2|
|Hyp 5|5 (per-fly + interaction)|4 (coupled HMM/GRU)|3 (factorization may amplify noise)|4 (structured decode)|4 (high)|4.0|

## Selection Decision
Primary: Hypothesis 2 (Hierarchical Semi-Markov VAE).

- Optimal for 30 Hz data because it explicitly models bout durations (semi-Markov) to counter frame-level over-segmentation, and separates slow (behavior identity) from fast (micro-kinematics) timescales. Hierarchy matches ethological structure (bouts composed of micro-movements).
- Balances segmentation vs generation by using discrete segment codes for interpretability and continuous/AR within-segment dynamics for fluid synthesis.
- Anticipated failure modes: (1) State proliferation/degeneracy → monitor effective number of states, entropy of posterior, and KL collapse; (2) Overly long segments (under-segmentation) → check duration distributions vs empirical bout lengths; (3) Within-segment kinematics too stiff → evaluate spectrum of Δpose and compare to real; (4) Transition overfitting → validate log-likelihood on held-out sequences and check transition sparsity.
- Alternatives: Hypothesis 4 (if we need higher generative fidelity via diffusion), Hypothesis 5 (if social coupling proves crucial).
