# Quick Start: Continuous vs Discrete Behavior Tokenization

This guide shows how to quickly train and compare continuous vs discrete approaches for behavior tokenization.

## What's New?

You now have **three approaches** to choose from:

1. **VQ-VAE (Discrete)** - Your current approach ✅ Already working
2. **VAE (Continuous)** - New continuous latent space approach
3. **Transformer/LSTM Forecasting** - Direct continuous-to-continuous prediction
4. **Hybrid** - Train continuous model → extract discrete tokens

## Prerequisites

```bash
# Make sure you have the data prepared
cd flies/data
# Your fly_keypoints.npy and fly_split.json should be ready
```

## Quick Training Commands

### 1. Train VQ-VAE (Discrete) - Baseline

```bash
cd flies/training
bash train_fixed.sh  # Your existing script
```

**Output**: Discrete behavior codes (0-511), saved to `outputs/vqvae_fixed/`

### 2. Train VAE (Continuous)

```bash
cd flies/training
python train_unified.py \
    --model_type vae \
    --config ../configs/vae_continuous.yaml \
    --output_dir outputs/vae_continuous \
    --use_wandb  # Optional: for logging
```

**Output**: Continuous latent representations, saved to `outputs/vae_continuous/`

**Variants to try:**
```bash
# β-VAE for more disentangled representations
python train_unified.py --model_type beta_vae --config ../configs/vae_continuous.yaml --output_dir outputs/beta_vae

# Annealed VAE for more stable training
python train_unified.py --model_type annealed_vae --config ../configs/vae_continuous.yaml --output_dir outputs/annealed_vae
```

### 3. Train Transformer Forecaster (Continuous-to-Continuous)

```bash
cd flies/training
python train_unified.py \
    --model_type transformer \
    --config ../configs/transformer_forecaster.yaml \
    --output_dir outputs/transformer_forecaster \
    --use_wandb
```

**Output**: Model that predicts future 75 frames from past 75 frames

**Note**: Transformer is more expensive! Use smaller model if needed:
```yaml
# Edit configs/transformer_forecaster.yaml
d_model: 128  # Instead of 256
num_layers: 4  # Instead of 6
```

## Quick Comparison

After training at least 2 models:

```python
cd flies/examples
python full_comparison_workflow.py \
    --data_dir ../data/fly_data \
    --vqvae_checkpoint ../training/outputs/vqvae_fixed/best_model.pt \
    --vae_checkpoint ../training/outputs/vae_continuous/best_model.pt \
    --output_dir ../comparison_results
```

This will generate:
- Reconstruction quality metrics
- Representation quality analysis
- Behavioral dynamics comparison
- Visualizations and report

## Hybrid Approach: Continuous → Discrete Tokens

```python
# In Python/Jupyter notebook
import sys
sys.path.append('flies')

from vq_vae.vae_continuous import ContinuousVAE
from hybrid.discrete_from_continuous import DiscreteTokenExtractor
from data.dataset import FlyKeypointDataset
from torch.utils.data import DataLoader
import torch

# 1. Load trained VAE
checkpoint = torch.load('flies/training/outputs/vae_continuous/best_model.pt')
vae = ContinuousVAE(
    input_dim=48,
    hidden_dims=[64, 128, 256],
    latent_dim=128,
    num_residual_blocks=2,
    sequence_length=150,
)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

# 2. Create token extractor
extractor = DiscreteTokenExtractor(
    continuous_model=vae,
    num_clusters=512,  # Match VQ-VAE codebook size
    clustering_method='minibatch_kmeans',
)

# 3. Extract continuous latents and cluster
train_dataset = FlyKeypointDataset(
    data_file='flies/data/fly_data/fly_keypoints.npy',
    fly_split_file='flies/data/fly_data/fly_split.json',
    split_name='train',
    window_size=150,
    stride=150,
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# Fit and extract tokens
tokens = extractor.fit_and_encode(train_loader, device='cuda')

# 4. Save for later
extractor.save('flies/outputs/vae_to_discrete_tokens.pkl')

print(f"Extracted {len(tokens)} discrete tokens")
print(f"Unique tokens used: {len(set(tokens))}/512")
```

**Now compare:** Do these tokens (from continuous model) capture richer behavior semantics than direct VQ-VAE tokens?

## What to Expect

### Training Time

| Model | Time per epoch (4025 train flies) | GPU Memory |
|-------|----------------------------------|------------|
| VQ-VAE | ~2-3 min | ~2 GB |
| VAE | ~2-3 min | ~2 GB |
| Transformer | ~10-15 min | ~6-8 GB |
| LSTM | ~5-7 min | ~3-4 GB |

### Convergence

- **VQ-VAE**: Should converge in 50-100 epochs, VQ loss < 10
- **VAE**: Converges faster, ~30-50 epochs, watch KL divergence
- **Transformer**: Needs 100+ epochs, watch forecast MSE

### Results Preview

**Reconstruction quality (expected MSE on test set):**
- VQ-VAE: ~0.001 - 0.005 (depends on codebook size)
- VAE: ~0.0005 - 0.002 (should be better due to continuous latent)
- Transformer: N/A (forecasting, not reconstruction)

**Representation quality:**
- VQ-VAE: 50-90% codebook utilization
- VAE: Intrinsic dimensionality ~10-30 (out of 128)
- Transformer: Hidden states encode rich dynamics

## Common Issues & Solutions

### 1. Out of Memory (Transformer)

**Solution**: Reduce model size in config
```yaml
# configs/transformer_forecaster.yaml
d_model: 128  # Instead of 256
num_layers: 4  # Instead of 6
batch_size: 32  # Instead of 64
```

### 2. VAE Posterior Collapse (KL loss → 0)

**Solution**: Use annealed VAE
```bash
python train_unified.py --model_type annealed_vae ...
```

### 3. Slow Training

**Solution**: Use more workers
```yaml
# In config YAML
num_workers: 8  # Increase from 4
```

### 4. Clustering Takes Too Long (Hybrid Approach)

**Solution**: Use MiniBatchKMeans
```python
extractor = DiscreteTokenExtractor(
    continuous_model=vae,
    num_clusters=512,
    clustering_method='minibatch_kmeans',  # Much faster!
)
```

## File Structure Reference

```
flies/
├── vq_vae/
│   ├── vqvae.py                    # VQ-VAE (discrete) ✅
│   ├── vae_continuous.py           # VAE (continuous) ✨ NEW
│   └── ...
├── forecasting/                     ✨ NEW
│   └── continuous_forecaster.py    # Transformer, LSTM
├── hybrid/                          ✨ NEW
│   └── discrete_from_continuous.py # Extract discrete from continuous
├── training/
│   ├── train.py                    # Original VQ-VAE training ✅
│   ├── train_unified.py            # Train all model types ✨ NEW
│   └── train_fixed.sh              # VQ-VAE training script ✅
├── evaluation/                      ✨ NEW
│   └── compare_models.py           # Comparison framework
├── configs/                         ✨ NEW
│   ├── vae_continuous.yaml
│   └── transformer_forecaster.yaml
├── examples/                        ✨ NEW
│   └── full_comparison_workflow.py
└── CONTINUOUS_VS_DISCRETE_GUIDE.md  ✨ NEW (comprehensive guide)
```

## Next Steps

1. **Train at least 2 models** (VQ-VAE + VAE recommended for quick comparison)
2. **Run comparison** using `examples/full_comparison_workflow.py`
3. **Analyze results**: Which approach is better for your research?
4. **Iterate**: Try β-VAE for disentanglement, or hybrid approach for best of both worlds

## Key Research Questions to Answer

1. **Reconstruction**: Does continuous approach (VAE) reconstruct better than discrete (VQ-VAE)?
2. **Dynamics**: Do discrete codes preserve behavioral dynamics (transition structure)?
3. **Hybrid**: Do tokens from continuous→discrete have richer semantics than direct VQ-VAE?
4. **Interpretability**: Are discrete codes easier to interpret than continuous latents?
5. **Scientific utility**: Which approach better identifies behavioral changes under perturbations?

## Need Help?

- 📖 **Detailed guide**: See `CONTINUOUS_VS_DISCRETE_GUIDE.md`
- 💻 **Code documentation**: Each file has detailed docstrings
- 🐛 **Issues**: Check existing VQ-VAE documentation (VQVAE.md, TRAIN.md)

## OpenAI Voice Model Connection

You mentioned OpenAI's voice models do audio→audio without discretization. That's analogous to:

- **VQ-VAE**: Discretizes behavior (like VQ-VAE on audio tokens)
- **VAE/Transformer**: Continuous latent dynamics (like continuous audio models)
- **Hybrid**: Get best of both - continuous dynamics + discrete tokens

Your research explores this same trade-off in the behavior domain! 🎯
