# Continuous vs Discrete Behavior Tokenization: A Comprehensive Guide

## Overview

This guide covers two fundamentally different approaches to learning behavior representations from continuous keypoint trajectories:

1. **Discrete Tokenization (VQ-VAE)**: Learn discrete "behavior syllables" via vector quantization
2. **Continuous Modeling (VAE/Forecasters)**: Learn continuous latent dynamics without discretization
3. **Hybrid Approach**: Train continuous model first, then extract discrete tokens

## Conceptual Differences

### Discrete Approach (VQ-VAE)

```
Keypoints (continuous) → Encoder → Continuous latent → [DISCRETIZATION] → Discrete codes → Decoder → Reconstructed keypoints
```

**Properties:**
- ✅ Creates interpretable behavior vocabulary (e.g., 64-512 distinct behaviors)
- ✅ Natural for Markov models, behavioral grammars, motif discovery
- ✅ Compression: Reduces continuous data to discrete symbols
- ❌ Information bottleneck: Forces all variations into fixed number of codes
- ❌ Hard assignment: Similar behaviors must choose one code

**When to use:**
- Want interpretable behavior "alphabet"
- Downstream tasks require discrete states (HMMs, n-gram models)
- Need to compare behavioral sequences between conditions
- Care more about categorical behavior types than subtle variations

### Continuous Approach (VAE)

```
Keypoints (continuous) → Encoder → μ, σ (latent distribution) → [SAMPLING] → Continuous latent → Decoder → Reconstructed keypoints
```

**Properties:**
- ✅ Preserves continuous variations within behavior types
- ✅ Smooth latent space: Similar behaviors → similar latents
- ✅ Generative: Can sample novel behaviors, interpolate
- ✅ No information bottleneck (except latent dimensionality)
- ❌ Less interpretable: Latent dimensions may be entangled
- ❌ Requires clustering post-hoc for discrete analysis

**When to use:**
- Want to capture subtle behavioral variations
- Need smooth interpolation between behaviors
- Generative modeling (sample new behaviors)
- Latent space should reflect continuous dynamics

### Continuous-to-Continuous Forecasting (Transformer/LSTM)

```
Past keypoints → Transformer/LSTM → Future keypoints (direct prediction, no reconstruction)
```

**Properties:**
- ✅ Learns temporal dynamics directly
- ✅ No reconstruction bottleneck
- ✅ Can be very expressive (large models)
- ✅ State-of-the-art for sequence modeling
- ❌ No explicit latent representation (unless extracted from hidden states)
- ❌ Expensive (transformers scale quadratically with sequence length)

**When to use:**
- Forecasting is the primary goal
- Have large datasets
- Want to learn complex temporal dependencies
- Can extract latents from intermediate layers for analysis

### Hybrid: Continuous → Discrete Token Extraction

```
Keypoints → Continuous model (VAE/Forecaster) → Continuous latents → [CLUSTERING] → Discrete tokens
```

**Rationale:**
> "Train model first and then extract discrete tokens from the model directly.
> The discrete representations/tokens we extract from the model will have richer
> representations of behavior that we can then input to VQ-VAE to get behavior codes."

**Properties:**
- ✅ **Best of both worlds**: Rich continuous dynamics + discrete interpretability
- ✅ Latents learned from dynamics, not just reconstruction
- ✅ Can use more powerful continuous model first
- ✅ Discrete tokens may be more semantically meaningful
- ❌ Two-stage process (train continuous, then cluster)
- ❌ Clustering quality depends on continuous model

**When to use:**
- Want discrete tokens but VQ-VAE alone is insufficient
- Believe continuous dynamics are important for learning good representations
- Have computational budget for two-stage approach
- Want to compare: (1) direct VQ-VAE vs (2) continuous → discrete

---

## Implementation Guide

### 1. Train Discrete Model (VQ-VAE)

**Already implemented!** You have a working VQ-VAE with normalization fix.

```bash
# Train VQ-VAE (current approach)
cd flies/training
bash train_fixed.sh
```

**What you get:**
- Discrete behavior codes (0 to num_embeddings-1)
- Each 150-frame window → 5 discrete codes (30× temporal compression)
- Codebook embeddings represent behavior "syllables"

### 2. Train Continuous Model (VAE)

```bash
# Train continuous VAE
python train_continuous.py \
    --model_type vae \
    --config ../configs/vae_continuous.yaml \
    --output_dir outputs/vae_continuous \
    --use_wandb
```

**Variants:**
- `--model_type vae`: Standard VAE (β=1)
- `--model_type beta_vae`: β-VAE for disentanglement (β=4-10)
- `--model_type annealed_vae`: KL annealing for stable training

**What you get:**
- Continuous latent codes (e.g., 128-dim vector per timestep)
- Smooth latent space (can interpolate between behaviors)
- Probabilistic representations (μ, σ)

### 3. Train Forecasting Model (Continuous-to-Continuous)

```bash
# Train transformer forecaster
python train_continuous.py \
    --model_type transformer \
    --config ../configs/transformer_forecaster.yaml \
    --output_dir outputs/transformer_forecaster \
    --use_wandb

# Or LSTM forecaster (faster, simpler)
python train_continuous.py \
    --model_type lstm \
    --config ../configs/lstm_forecaster.yaml \
    --output_dir outputs/lstm_forecaster
```

**What you get:**
- Model that predicts future keypoints from past
- Learns temporal dynamics end-to-end
- Can extract latents from encoder for analysis

### 4. Hybrid Approach: Extract Discrete Tokens from Continuous Model

```python
# After training VAE
from hybrid.discrete_from_continuous import DiscreteTokenExtractor
from torch.utils.data import DataLoader

# Load trained continuous model
vae_model = load_model('outputs/vae_continuous/best_model.pt')

# Create token extractor
extractor = DiscreteTokenExtractor(
    continuous_model=vae_model,
    num_clusters=512,  # Same as VQ-VAE for fair comparison
    clustering_method='kmeans',
    use_pca=True,  # Optional: reduce dimensionality before clustering
    pca_dims=32,
)

# Extract continuous latents and cluster
train_dataloader = DataLoader(...)
tokens = extractor.fit_and_encode(train_dataloader, device='cuda')

# Save for later use
extractor.save('outputs/vae_to_discrete/token_extractor.pkl')

# Now you have discrete tokens from continuous model!
# Compare to direct VQ-VAE tokens
```

**What you get:**
- Discrete tokens (like VQ-VAE) but learned from continuous dynamics
- Potentially richer behavioral semantics
- Can compare: VQ-VAE tokens vs VAE→clustered tokens

---

## Comparison Framework

### Run Full Comparison

```python
from evaluation.compare_models import run_full_comparison
from data.dataset import FlyKeypointDataset
from torch.utils.data import DataLoader

# Load all trained models
vqvae = load_model('outputs/vqvae/best_model.pt')
vae = load_model('outputs/vae_continuous/best_model.pt')
transformer = load_model('outputs/transformer_forecaster/best_model.pt')

# Prepare test data
test_dataset = FlyKeypointDataset(...)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Run comparison
model_configs = [
    ('VQ-VAE (discrete)', vqvae, 'vqvae'),
    ('VAE (continuous)', vae, 'vae'),
    ('Transformer (continuous)', transformer, 'transformer'),
]

comparator = run_full_comparison(
    model_configs=model_configs,
    dataloader=test_loader,
    output_dir='comparison_results',
)
```

### Metrics Compared

**1. Reconstruction Quality**
- MSE (overall and per-keypoint)
- Velocity error (1st derivative)
- Acceleration error (2nd derivative)

**2. Representation Quality**
- **Discrete models**: Codebook utilization, code entropy
- **Continuous models**: Intrinsic dimensionality, clustering quality (silhouette score)

**3. Behavioral Dynamics**
- **Discrete**: Transition matrices, transition entropy
- **Continuous**: Temporal autocorrelation, latent space structure

---

## Expected Results & Insights

### Reconstruction Quality

**Hypothesis**: Continuous models should have **lower MSE** due to no discrete bottleneck.

| Model | Expected MSE | Reason |
|-------|-------------|--------|
| VQ-VAE | Moderate | Discrete bottleneck limits reconstruction |
| VAE | Lower | Continuous latent space, more expressive |
| Transformer | Lowest (on forecasting) | No reconstruction bottleneck, direct prediction |

**BUT**: Lower MSE ≠ better representations! Discrete models may learn more interpretable structure.

### Representation Quality

**Hypothesis**: Different models learn different aspects of behavior.

**VQ-VAE:**
- Should use 50-90% of codebook (good utilization)
- High code entropy = diverse behaviors
- Codes represent categorical behavior types

**VAE:**
- Intrinsic dimensionality << latent_dim (most dimensions unused)
- β-VAE should have better disentanglement
- Smooth latent space (good for interpolation)

**Transformer:**
- Hidden states encode rich temporal context
- May be harder to interpret
- Best for forecasting, not necessarily for analysis

### Behavioral Dynamics

**Key question**: Does discretization preserve behavioral dynamics?

**Test**: Compare transition statistics
- Extract transitions from VQ-VAE codes
- Cluster VAE latents → extract transitions
- Compare transition matrices

**Hypothesis**: If discretization preserves dynamics, VQ-VAE and VAE→cluster should have similar transition structures.

---

## Scientific Interpretation for Your Field

### For Neuroscience/Animal Behavior Research

**Discrete Approach (VQ-VAE):**
- Natural for hypothesis: "Flies have distinct behavioral states"
- Easy to communicate: "We identified 64 behavior syllables"
- Enables comparisons: "Mutant X shows more grooming behaviors (code 12)"

**Continuous Approach (VAE):**
- Natural for hypothesis: "Behavior is a continuous process"
- Captures variability: "Within 'walking', flies vary in speed and direction"
- Better for correlations: "Behavior correlates with neural activity in continuous space"

**Hybrid Approach:**
- Best for: "Extract discrete behaviors, but ensure they capture dynamics"
- Validation: "Our discrete codes preserve temporal structure of continuous behavior"

### Framing for Your Research

> **Novelty claim**: "We show that discrete behavior tokens extracted from continuous
> dynamics models preserve temporal structure better than direct tokenization,
> enabling both interpretable analysis AND forecasting."

**Experiment design:**
1. Train all models on same data
2. Compare reconstruction & forecasting
3. Extract discrete codes from all models (VQ-VAE direct, VAE→cluster, Transformer→cluster)
4. Analyze: Do codes preserve dynamics? (transition stats, Markov models)
5. **Key test**: "Given we stimulate/change X, we observed changes in behaviors Y"
   - Do all methods identify same behavior changes?
   - Which method is most sensitive?
   - Which method is most interpretable?

---

## Quick Start Example

```bash
# 1. Train all models (in parallel if you have GPUs)
cd flies/training

# VQ-VAE (baseline - already trained)
bash train_fixed.sh

# VAE (continuous)
python train_continuous.py --model_type vae --config ../configs/vae_continuous.yaml --output_dir outputs/vae

# Transformer (continuous-to-continuous)
python train_continuous.py --model_type transformer --config ../configs/transformer_forecaster.yaml --output_dir outputs/transformer

# 2. Compare models
cd ../evaluation
python -c "
from compare_models import run_full_comparison
# ... (load models and run comparison)
"

# 3. Extract discrete tokens from continuous models (hybrid approach)
python -c "
from hybrid.discrete_from_continuous import DiscreteTokenExtractor
# ... (extract and compare tokens)
"
```

---

## Next Steps

1. **Train all models** on your fruit fly data
2. **Run comparison** using the provided framework
3. **Analyze results**:
   - Which model has best reconstruction?
   - Which has most interpretable representations?
   - Does hybrid approach (continuous→discrete) outperform direct VQ-VAE?
4. **Scientific validation**:
   - Do discrete codes identify known behaviors (grooming, walking, etc.)?
   - Can you forecast future behavior accurately?
   - Do models identify same behavioral changes under perturbations?

## Questions to Answer with This Framework

1. **Information preservation**: How much information is lost in discretization?
   - Compare VAE (continuous) vs VQ-VAE (discrete) reconstruction

2. **Dynamics preservation**: Does discretization preserve temporal structure?
   - Compare transition matrices: VQ-VAE vs VAE→cluster

3. **Interpretability**: Are discrete codes more interpretable?
   - Visualize codebook entries (what behavior does each code represent?)
   - Compare to continuous latent space (what do dimensions mean?)

4. **Generalization**: Which approach generalizes better to new data?
   - Train on one cohort, test on another
   - Which model maintains performance?

5. **Scientific utility**: Which is better for your research questions?
   - Discrete: "Which behaviors change under condition X?"
   - Continuous: "How does behavior vary continuously with stimulus intensity?"

---

## File Structure

```
flies/
├── vq_vae/
│   ├── vqvae.py                    # Discrete VQ-VAE (current)
│   ├── vae_continuous.py           # Continuous VAE (NEW)
│   ├── quantizer.py                # Vector quantizer
│   └── ...
├── forecasting/
│   └── continuous_forecaster.py    # Transformer/LSTM (NEW)
├── hybrid/
│   └── discrete_from_continuous.py # Extract tokens from continuous (NEW)
├── training/
│   ├── train.py                    # Original VQ-VAE training
│   ├── train_continuous.py            # Unified training for all models (NEW)
│   └── ...
├── evaluation/
│   └── compare_models.py           # Comparison framework (NEW)
├── configs/
│   ├── vae_continuous.yaml         # VAE config (NEW)
│   └── transformer_forecaster.yaml # Transformer config (NEW)
└── CONTINUOUS_VS_DISCRETE_GUIDE.md # This file
```

---

## References & Related Work

**VQ-VAE for behavior**:
- Berman et al. 2014 - Mapping the stereotyped behaviour of freely moving fruit flies
- Wiltschko et al. 2015 - Mapping sub-second structure in mouse behavior

**Continuous latent dynamics**:
- Pandarinath et al. 2018 - Inferring single-trial neural population dynamics using sequential auto-encoders

**Hybrid approaches**:
- Your work! This is novel for behavior analysis.

**Comparison philosophy**:
- Scientists love comparisons - showing both approaches and their trade-offs will make your paper stronger.
