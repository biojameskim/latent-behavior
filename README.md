# Latent Behavior Analysis - Fruit Flies

This repository implements multiple approaches for learning behavior representations from continuous keypoint trajectories of fruit flies, comparing **discrete** vs **continuous** tokenization strategies.

## 📖 Overview

**Research Question**: How should we tokenize continuous behavior data - discrete codes (VQ-VAE) or continuous latent spaces (VAE)?

This codebase implements and compares:
1. **Discrete tokenization (VQ-VAE)** - Learn discrete "behavior syllables"
2. **Continuous latent spaces (VAE)** - Preserve continuous dynamics
3. **Continuous forecasting (Transformer/LSTM)** - Direct prediction without reconstruction
4. **Hybrid approach** - Train continuous model → extract discrete tokens

## 🚀 Quick Start

### Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate lat-beh
```

### Train Models

```bash
cd flies/training

# 1. VQ-VAE (discrete tokenization)
bash train_fixed.sh

# 2. VAE (continuous latent space)
python train_unified.py --model_type vae \
    --config ../configs/vae_continuous.yaml \
    --output_dir outputs/vae_continuous

# 3. Transformer forecaster (continuous-to-continuous)
python train_unified.py --model_type transformer \
    --config ../configs/transformer_forecaster.yaml \
    --output_dir outputs/transformer_forecaster
```

### Compare Approaches

```bash
cd flies/examples
python full_comparison_workflow.py \
    --data_dir ../data/fly_data \
    --vqvae_checkpoint ../training/outputs/vqvae_fixed/best_model.pt \
    --vae_checkpoint ../training/outputs/vae_continuous/best_model.pt \
    --output_dir ../comparison_results
```

## 📁 Project Structure

```
latent-behavior/
├── README.md                           # This file
├── QUICKSTART_CONTINUOUS.md            # Quick tutorial for continuous approaches
├── environment.yml                     # Conda environment specification
│
├── flies/                              # Main codebase
│   ├── CONTINUOUS_VS_DISCRETE_GUIDE.md # Comprehensive conceptual guide
│   ├── fly_info.md                     # Dataset and fly keypoint documentation
│   │
│   ├── data/                           # Data loading and preprocessing
│   │   ├── DATASET.md                  # Dataset documentation
│   │   ├── preprocessing.py            # Load and preprocess keypoints
│   │   ├── dataset.py                  # PyTorch Dataset class
│   │   ├── prepare_data.py             # Data preparation utilities
│   │   └── create_train_val_split.py   # Create train/val splits
│   │
│   ├── vq_vae/                         # Model implementations
│   │   ├── VQVAE.md                    # VQ-VAE documentation
│   │   ├── vqvae.py                    # VQ-VAE (discrete) - main implementation ⭐
│   │   ├── vqvae_simple.py             # VQ-VAE with minimal fix (alternative)
│   │   ├── vae_continuous.py           # VAE (continuous) - NEW ⭐
│   │   ├── seq_encoder.py              # 1D convolutional encoder
│   │   ├── seq_decoder.py              # 1D convolutional decoder
│   │   ├── quantizer.py                # Vector quantizer (with normalization fix)
│   │   ├── quantizer_simple.py         # Vector quantizer (minimal fix)
│   │   └── residual.py                 # Residual blocks
│   │
│   ├── forecasting/                    # Continuous-to-continuous models - NEW ⭐
│   │   ├── __init__.py
│   │   └── continuous_forecaster.py    # Transformer & LSTM forecasters
│   │
│   ├── hybrid/                         # Hybrid discrete-continuous approaches - NEW ⭐
│   │   ├── __init__.py
│   │   └── discrete_from_continuous.py # Extract discrete tokens from continuous models
│   │
│   ├── training/                       # Training scripts and utilities
│   │   ├── TRAIN.md                    # Training documentation
│   │   ├── NORMALIZATION_GUIDE.md      # Guide to normalization fix
│   │   ├── train.py                    # Original VQ-VAE training script
│   │   ├── train_unified.py            # Unified training for all models - NEW ⭐
│   │   ├── train_fixed.sh              # VQ-VAE training (with normalization)
│   │   ├── train_simple.sh             # VQ-VAE training (minimal fix)
│   │   ├── inspect_checkpoint.py       # Checkpoint inspection utility
│   │   └── debug_*.py                  # Debugging utilities
│   │
│   ├── evaluation/                     # Model comparison and evaluation - NEW ⭐
│   │   ├── __init__.py
│   │   └── compare_models.py           # Comprehensive comparison framework
│   │
│   ├── examples/                       # Example workflows - NEW ⭐
│   │   └── full_comparison_workflow.py # Complete comparison pipeline
│   │
│   ├── visualization/                  # Visualization tools
│   │   ├── VISUALIZATION.md            # Visualization documentation
│   │   ├── visualize_reconstructions.py # Compare original vs reconstructed
│   │   ├── visualize_codebook_embeddings.py # Decode codebook entries
│   │   ├── reconstruction.py           # Reconstruction utilities
│   │   ├── create_video.py             # Create behavior videos
│   │   └── plot_mabe_flies.py          # Plot fly keypoints
│   │
│   └── configs/                        # Configuration files - NEW ⭐
│       ├── vae_continuous.yaml         # VAE configuration
│       └── transformer_forecaster.yaml # Transformer configuration
│
└── notebooks/                          # Jupyter notebooks (if any)
```

## 🔬 Approaches Implemented

### 1. VQ-VAE (Discrete Tokenization) ✅

**Location**: `flies/vq_vae/vqvae.py`

**What it does**:
- Learns discrete "behavior syllables" (e.g., 512 codes)
- Each 5-second window → 5 discrete behavior codes
- Creates interpretable behavior vocabulary

**Use when**:
- Need discrete states for Markov models, behavioral grammars
- Want interpretable behavior categories
- Care about categorical behavior types over subtle variations

**Documentation**: `flies/vq_vae/VQVAE.md`, `flies/training/TRAIN.md`

### 2. VAE (Continuous Latent Space) ⭐ NEW

**Location**: `flies/vq_vae/vae_continuous.py`

**What it does**:
- Learns continuous latent space (no discrete bottleneck)
- Preserves subtle behavioral variations
- Smooth interpolation between behaviors

**Variants**:
- `ContinuousVAE`: Standard VAE (β=1)
- `BetaVAE`: β-VAE for disentanglement (β=4-10)
- `AnnealedVAE`: KL annealing for stable training

**Use when**:
- Want to capture continuous variations in behavior
- Need generative modeling (sample/interpolate behaviors)
- Latent space should reflect continuous dynamics

**Documentation**: `flies/CONTINUOUS_VS_DISCRETE_GUIDE.md`, `QUICKSTART_CONTINUOUS.md`

### 3. Continuous Forecasting (Transformer/LSTM) ⭐ NEW

**Location**: `flies/forecasting/continuous_forecaster.py`

**What it does**:
- Predicts future keypoints from past keypoints directly
- No reconstruction bottleneck
- Learns temporal dynamics end-to-end

**Use when**:
- Forecasting is the primary goal
- Want to model complex temporal dependencies
- Can extract latents from hidden states for analysis

**Documentation**: `flies/CONTINUOUS_VS_DISCRETE_GUIDE.md`

### 4. Hybrid: Continuous → Discrete ⭐ NEW

**Location**: `flies/hybrid/discrete_from_continuous.py`

**What it does**:
- Train continuous model (VAE) first
- Extract discrete tokens via clustering (K-means)
- Get discrete interpretability + continuous dynamics

**Key idea**: Discrete tokens from continuous models may have richer behavioral semantics than direct VQ-VAE

**Use when**:
- Want best of both worlds (continuous dynamics + discrete tokens)
- Believe continuous learning helps representation quality
- Need discrete tokens for downstream analysis

**Documentation**: `flies/CONTINUOUS_VS_DISCRETE_GUIDE.md`

## 📊 Comparison Framework

**Location**: `flies/evaluation/compare_models.py`

Evaluates all approaches on:

1. **Reconstruction Quality**
   - MSE (overall and per-keypoint)
   - Velocity error (temporal consistency)
   - Acceleration error (dynamics preservation)

2. **Representation Quality**
   - Codebook utilization (discrete models)
   - Intrinsic dimensionality (continuous models)
   - Clustering quality (silhouette score, Davies-Bouldin)

3. **Behavioral Dynamics**
   - Transition matrices (discrete codes)
   - Temporal autocorrelation (continuous latents)
   - Dynamics preservation metrics

**Usage**: See `flies/examples/full_comparison_workflow.py`

## 📚 Documentation

- **Quick Start**: `QUICKSTART_CONTINUOUS.md` - Get started quickly
- **Conceptual Guide**: `flies/CONTINUOUS_VS_DISCRETE_GUIDE.md` - Deep dive into approaches
- **VQ-VAE Details**: `flies/vq_vae/VQVAE.md` - Architecture and implementation
- **Training Guide**: `flies/training/TRAIN.md` - Training tips and hyperparameters
- **Dataset Info**: `flies/data/DATASET.md` - Data format and preprocessing
- **Visualization**: `flies/visualization/VISUALIZATION.md` - Visualization tools

## 🎯 Key Research Questions

1. **Reconstruction**: Does continuous approach (VAE) reconstruct better than discrete (VQ-VAE)?
2. **Dynamics**: Does discretization preserve behavioral dynamics (transition structure)?
3. **Hybrid**: Do tokens from continuous→discrete have richer semantics than direct VQ-VAE?
4. **Interpretability**: Are discrete codes easier to interpret than continuous latents?
5. **Scientific utility**: Which approach better identifies behavioral changes under perturbations?

## 🔧 Key Features

### Discrete Approach (VQ-VAE)
✅ Pre-quantizer normalization fix (prevents codebook collapse)
✅ Automatic stride computation for any sequence length
✅ Rotation-invariant canonical reference frame
✅ Comprehensive visualization tools

### Continuous Approaches (NEW)
⭐ Unified training framework for all model types
⭐ Comprehensive comparison metrics
⭐ Hybrid token extraction from continuous models
⭐ Multiple VAE variants (standard, β-VAE, annealed)
⭐ Transformer & LSTM forecasters

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{latent_behavior_flies,
  title = {Latent Behavior Analysis: Discrete vs Continuous Tokenization},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/latent-behavior}
}
```

## 🙏 Acknowledgments

- Branson Lab's [FlyLLM](https://github.com/kristinbranson/AnimalPoseForecasting/tree/c5d61ac2ee6109287d104aeb4465858b1eae603f/flyllm)
- MABe 2022 Challenge for the fruit fly dataset

## 📄 License

[Add your license here]

## 🐛 Issues & Contributing

For bugs or feature requests, please open an issue on GitHub.

---

**Latest Updates**:
- ⭐ Added continuous behavior modeling approaches (VAE, Transformer, LSTM)
- ⭐ Added hybrid discrete token extraction from continuous models
- ⭐ Added comprehensive comparison framework
- ✅ Fixed VQ-VAE codebook collapse with pre-quantizer normalization
- ✅ Added automatic stride computation for flexible sequence lengths
