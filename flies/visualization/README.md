# VQ-VAE Visualization Tools

This directory contains tools for visualizing VQ-VAE codebooks and reconstructions.

## Overview

After training a VQ-VAE model, these scripts help you:
1. **Understand what the codebook learned** - visualize each discrete behavior "syllable"
2. **Evaluate reconstruction quality** - compare original vs reconstructed trajectories
3. **Explore behavior composition** - generate custom behavior sequences

## Quick Start

### 1. Visualize All Codebook Embeddings

See what each discrete code represents:

```bash
python visualize_codebook_embeddings.py \
    --checkpoint ../training/outputs/my_run/best_model.pt \
    --output_dir codebook_viz \
    --frame_indices 0 74 149
```

**Output**: One image per codebook entry showing the behavior it represents.

### 2. Visualize Reconstructions on Real Data

Evaluate reconstruction quality:

```bash
python visualize_reconstructions.py \
    --data_file ../../data/fly_data/fly_group_train.npy \
    --checkpoint ../training/outputs/my_run/best_model.pt \
    --fly_split_file ../../data/fly_data/fly_split.json \
    --val_split_name val \
    --output_dir recon_viz
```

**Output**:
- Window overlays: Individual window comparisons (original vs reconstructed)
- Sequence overlays: Full trajectory comparisons

### 3. Tutorial: Understanding the Basics

Learn the core concepts step-by-step:

```bash
python tutorial_codebook_viz.py \
    --checkpoint ../training/outputs/my_run/best_model.pt \
    --output_dir tutorial_outputs
```

**Output**: Educational examples with detailed console output explaining each step.

## File Guide

### Core Visualization Scripts

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `visualize_codebook_embeddings.py` | Decode all codebook embeddings | Checkpoint | Images of each code's behavior |
| `visualize_reconstructions.py` | Evaluate reconstruction quality | Checkpoint + data | Original vs reconstructed comparisons |
| `tutorial_codebook_viz.py` | Learn the basics interactively | Checkpoint | Educational examples |

### Helper Modules

| File | Purpose |
|------|---------|
| `reconstruction.py` | Utilities for format conversion, windowing, plotting overlays |
| `plot_mabe_flies.py` | Low-level fly skeleton and keypoint rendering |
| `create_video.py` | Create videos from sequences |

### Documentation

| File | Contents |
|------|----------|
| `CODEBOOK_VISUALIZATION_GUIDE.md` | **📖 Comprehensive guide** - read this first! |
| `VISUALIZATION.md` | General visualization documentation |
| `README.md` | This file |

## Key Concepts

### What is a Codebook?

The VQ-VAE learns a **codebook** (vocabulary) of discrete embeddings:
- Each embedding = one learned behavior "syllable"
- Codebook size: typically 64, 128, or 512 embeddings
- Each embedding is a high-dimensional vector (e.g., 256-dim)

### How to Visualize a Codebook Entry?

**Core idea**: Decode it!

```python
# Example: What does code #42 represent?
code_sequence = [42, 42, 42, 42, 42]  # Repeat across latent timesteps
behavior = model.decode_codes(code_sequence)
# Now visualize the behavior!
```

This converts the discrete code → continuous keypoint trajectories → fly poses.

### Three Types of Visualization

1. **Individual embeddings** (pure behavior syllables)
   - Decode each code independently: `[0,0,0,0,0]`, `[1,1,1,1,1]`, etc.
   - Shows what each code "means" in isolation

2. **Reconstructions** (quality evaluation)
   - Encode real data → quantize → decode
   - Compare original vs reconstructed to assess quality

3. **Custom sequences** (behavior composition)
   - Create custom code sequences: `[12, 42, 17, 5, 28]`
   - Explore how behaviors combine and transition

## Understanding the Pipeline

```
┌─────────────┐
│ Input Data  │  (150 frames of fly keypoints)
└──────┬──────┘
       │
       ▼
   ┌────────┐
   │Encoder │  (CNN downsampling)
   └────┬───┘
        │
        ▼
  ┌───────────┐
  │ Quantizer │  (discrete codebook lookup)
  └─────┬─────┘
        │
        ▼ [12, 42, 17, 5, 28]  ← Discrete codes
        │
        ▼
   ┌────────┐
   │Decoder │  (CNN upsampling)
   └────┬───┘
        │
        ▼
┌──────────────┐
│Reconstructed │  (150 frames reconstructed)
└──────────────┘
```

**Codebook visualization** = Start from discrete codes, run through decoder.

## Common Use Cases

### "I want to understand what my model learned"

→ Run `visualize_codebook_embeddings.py`

Look at the output images:
- Do different codes show different behaviors?
- Are the behaviors semantically meaningful?
- Is the codebook diverse or collapsed?

### "I want to check reconstruction quality"

→ Run `visualize_reconstructions.py`

Check the overlays:
- Blue = original behavior
- Orange = reconstructed behavior
- Good reconstruction = close overlap

### "I want to generate new behaviors"

→ Use `visualize_codebook_embeddings.py --codes '[[12,42,17]]'`

Create custom sequences to compose behaviors from learned syllables.

### "I'm new and want to learn the basics"

→ Run `tutorial_codebook_viz.py` and read the console output

Then read `CODEBOOK_VISUALIZATION_GUIDE.md` for comprehensive understanding.

## Tips and Best Practices

### Choosing Frame Indices

For a 150-frame window, good defaults:
- `--frame_indices 0 74 149` (start, middle, end)
- `--frame_indices 0 49 99 149` (4 snapshots)

### Memory Management

Large codebooks (512+ embeddings) can use lots of memory:
- Use `--chunk_size 64` to decode in batches
- Use `--device cpu` if GPU memory is limited

### Interpreting Results

**Good signs**:
- ✅ Visually distinct behaviors across codes
- ✅ Smooth, natural-looking fly poses
- ✅ Semantically meaningful patterns (turning, walking, etc.)

**Bad signs**:
- ❌ Many codes look identical (codebook collapse)
- ❌ Unnatural or distorted poses (poor decoder)
- ❌ Reconstructions differ significantly from originals

## Troubleshooting

### "Checkpoint missing 'args' key"

The checkpoint doesn't contain training configuration. Ensure your training script saves:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'args': vars(args),  # ← Include this!
    ...
}, checkpoint_path)
```

### "All embeddings look the same"

**Codebook collapse** - model not using full codebook.

Solutions:
- Increase `commitment_cost` (e.g., 0.25 → 0.5)
- Use improved VQ with dead code removal (`UnifiedVQVAE` with `vq_improved`)
- Check training metrics (perplexity should be high)

### "Out of memory"

Reduce memory usage:
- `--chunk_size 32` (decode fewer embeddings at once)
- `--device cpu` (use CPU instead of GPU)
- Reduce `--dpi 100` (lower resolution images)

### "Poor reconstruction quality"

Model may be undertrained or underfitting:
- Train for more epochs
- Increase model capacity (`hidden_dims = [128, 256, 512]`)
- Check data preprocessing (normalization, windowing)
- Verify loss is decreasing during training

## Next Steps

1. ✅ Read `CODEBOOK_VISUALIZATION_GUIDE.md` for comprehensive understanding
2. ✅ Run `tutorial_codebook_viz.py` on a trained model
3. ✅ Run `visualize_codebook_embeddings.py` to see all learned behaviors
4. ✅ Run `visualize_reconstructions.py` to evaluate quality
5. ✅ Experiment with custom code sequences
6. ✅ Use insights for downstream analysis (behavior clustering, etc.)

## References

- **VQ-VAE Paper**: [Neural Discrete Representation Learning (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- **Main VQ-VAE Implementation**: `../vq_vae/vqvae.py`
- **Training Scripts**: `../training/train.py`

## Questions or Issues?

Check the documentation:
- `CODEBOOK_VISUALIZATION_GUIDE.md` - Comprehensive guide
- Inline comments in scripts
- Docstrings in functions

If you're still stuck, the issue is likely:
1. Checkpoint format (missing 'args')
2. Data format mismatch
3. Model architecture change

Debug by running `tutorial_codebook_viz.py` first - it has detailed error messages.
