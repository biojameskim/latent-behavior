# VQ-VAE Codebook Visualization Guide

## Table of Contents
1. [Overview](#overview)
2. [Conceptual Framework](#conceptual-framework)
3. [Three Types of Visualization](#three-types-of-visualization)
4. [Implementation Details](#implementation-details)
5. [Quick Start Examples](#quick-start-examples)
6. [Understanding the Output](#understanding-the-output)

---

## Overview

This guide explains **how to visualize your VQ-VAE's learned codebook** to understand what discrete behavior "syllables" your model has discovered.

### What is a Codebook?

The VQ-VAE learns a **codebook** (also called a "vocabulary") of discrete embeddings:
- Each embedding represents a prototypical temporal pattern
- For fly behavior, each embedding = one learned behavior "syllable"
- The codebook has `num_embeddings` entries (e.g., 64, 128, 512)

### Why Visualize the Codebook?

**Question**: What does code #42 actually represent in terms of behavior?

**Answer**: Decode it! Pass the embedding through the decoder to see what behavior it reconstructs.

---

## Conceptual Framework

### The VQ-VAE Pipeline

```
Input Behavior          Encoder        Quantizer              Decoder        Reconstructed
(keypoints)         →   (CNN)      →   (codebook)         →   (CNN)      →   Behavior
                                         ↓
                                    Discrete Codes
                                    [12, 42, 17, ...]
```

### Three Key Components

1. **Encoder**: Converts continuous keypoint trajectories → continuous latent representations
   - Input: `(batch, 48, 150)` - 150 frames of 24 keypoints (x,y)
   - Output: `(batch, embedding_dim, latent_length)` - compressed temporal representation
   - Example: `(batch, 256, 5)` - 150 frames compressed to 5 latent timesteps

2. **Quantizer (Codebook)**: Discretizes continuous latents → discrete code indices
   - Contains `num_embeddings` learned vectors (e.g., 64 or 512 embeddings)
   - Maps each latent vector to nearest codebook entry
   - Output: `(batch, latent_length)` - sequence of discrete codes
   - Example: `[12, 42, 17, 5, 28]` - 5 discrete behavior codes

3. **Decoder**: Converts quantized embeddings → reconstructed behavior
   - Input: `(batch, embedding_dim, latent_length)` - quantized latent vectors
   - Output: `(batch, 48, 150)` - reconstructed keypoint trajectories

---

## Three Types of Visualization

### 1. Visualizing Individual Codebook Embeddings

**Purpose**: Understand what each discrete code represents

**Method**:
```python
# For each code in codebook (e.g., code 42):
code_idx = 42
code_sequence = [42, 42, 42, 42, 42]  # Repeat across latent_length

# Decode to see what behavior this represents
reconstructed_behavior = model.decode_codes(code_sequence)

# Visualize the reconstructed fly pose
plot_fly_poses(reconstructed_behavior)
```

**Script**: `visualize_codebook_embeddings.py`

**Output**: One image per codebook entry showing what that behavior "syllable" looks like

**Key Insight**: This shows you the **pure form** of each learned behavior pattern

---

### 2. Visualizing Reconstructions from Real Data

**Purpose**: Evaluate reconstruction quality on actual fly trajectories

**Method**:
```python
# 1. Take real fly behavior
original_behavior = load_validation_data()

# 2. Run through full VQ-VAE
reconstructed, codes = model(original_behavior)

# 3. Compare original vs reconstructed
plot_overlay(original_behavior, reconstructed)
```

**Script**: `visualize_reconstructions.py`

**Output**: Side-by-side comparisons of original vs reconstructed behavior

**Key Insight**: Shows how well the discrete codebook can represent **real continuous behavior**

---

### 3. Visualizing Custom Code Sequences

**Purpose**: Explore what happens when you combine different behavior codes

**Method**:
```python
# Create custom sequence of codes
custom_sequence = [12, 42, 17, 5, 28]

# Decode to see what behavior this creates
behavior = model.decode_codes(custom_sequence)

# Visualize
plot_fly_poses(behavior)
```

**Script**: `visualize_codebook_embeddings.py --codes '[[12,42,17]]'`

**Output**: Behavior generated from your custom code sequence

**Key Insight**: Lets you **compose** behaviors from learned syllables

---

## Implementation Details

### Key Functions

#### 1. `model.decode_codes(encoding_indices)`
**Location**: `flies/vq_vae/vqvae.py:214-243`

**Purpose**: Decode discrete code indices → reconstructed behavior

**How it works**:
```python
def decode_codes(self, encoding_indices):
    """
    Args:
        encoding_indices: (batch_size, latent_length)
                         e.g., [[12, 42, 17, 5, 28]]

    Returns:
        x_recon: (batch_size, 48, 150) - reconstructed keypoints
    """
    # 1. Convert indices to one-hot vectors
    # encoding_indices [12, 42, ...] → one_hot encodings

    # 2. Lookup embeddings from codebook
    # one_hot × codebook.weight → quantized embeddings

    # 3. Pass through decoder
    # decoder(quantized) → reconstructed behavior
```

**Steps**:
1. **Index → One-hot**: Convert code indices to one-hot vectors
   - Code 42 → `[0, 0, ..., 1, ..., 0]` (1 at position 42)

2. **One-hot × Codebook**: Multiply by codebook weight matrix
   - Extracts the actual embedding vector for code 42
   - Shape: `(batch, latent_length, embedding_dim)`

3. **Reshape**: Convert to `(batch, embedding_dim, latent_length)` for decoder

4. **Decode**: Pass through decoder CNN
   - Upsamples latent representation back to full resolution
   - Output: reconstructed keypoint trajectories

#### 2. `window_to_pose(window)`
**Location**: `flies/visualization/reconstruction.py:26-50`

**Purpose**: Convert flat feature vectors → structured pose format

**Transformation**:
```
Input:  (48, 150) or (150, 48)
        ↓
Output: (150, 24, 2)
        [time, keypoint, (x,y)]
```

**Why needed**: Model works with flat vectors `(48, T)`, but visualization needs structured poses `(T, 24, 2)`

#### 3. `plot_fly(pose, ax, ...)`
**Location**: `flies/visualization/plot_mabe_flies.py`

**Purpose**: Render a fly skeleton with keypoints

**Input**: `(24, 2)` array - one frame of keypoint positions

**Output**: Matplotlib plot showing fly pose with skeleton connections

---

## Quick Start Examples

### Example 1: Visualize All Codebook Embeddings

```bash
cd flies/visualization

python visualize_codebook_embeddings.py \
    --checkpoint ../training/outputs/my_run/best_model.pt \
    --output_dir codebook_viz \
    --frame_indices 0 74 149
```

**Output**: `codebook_viz/codebook_embeddings/embedding_0000.png` through `embedding_NNNN.png`

**What you'll see**:
- One image per codebook entry
- Each image shows 3 frames (start, middle, end) of the behavior that code represents
- Helps you understand: "Code 42 = left turn", "Code 17 = walking forward", etc.

---

### Example 2: Visualize Reconstructions on Validation Data

```bash
python visualize_reconstructions.py \
    --data_file ../data/fly_data/fly_group_train.npy \
    --checkpoint ../training/outputs/my_run/best_model.pt \
    --fly_split_file ../data/fly_data/fly_split.json \
    --val_split_name val \
    --output_dir recon_viz
```

**Output**:
- `recon_viz/window_overlays/` - Individual window comparisons
- `recon_viz/sequence_overlays/` - Full trajectory comparisons

**What you'll see**:
- Blue = original behavior
- Orange = reconstructed behavior
- Shows how well discrete codes can approximate continuous behavior

---

### Example 3: Visualize Custom Code Sequences

```bash
python visualize_codebook_embeddings.py \
    --checkpoint ../training/outputs/my_run/best_model.pt \
    --output_dir custom_viz \
    --codes '[[12, 42, 17, 5, 28], [42, 42, 42, 42, 42], [1, 2, 3, 4, 5]]'
```

**Output**: `custom_viz/custom_sequences/custom_01.png`, `custom_02.png`, etc.

**What you'll see**:
- Sequence 1: Mixed behavior composed of codes 12→42→17→5→28
- Sequence 2: Pure code 42 repeated (same as embedding_0042.png)
- Sequence 3: Progression through codes 1→2→3→4→5

---

## Understanding the Output

### Interpreting Codebook Embeddings

**Good signs**:
- ✅ Each code shows visually distinct behavior
- ✅ Similar codes show similar behaviors (local coherence)
- ✅ Codes cover diverse behaviors (walking, turning, grooming, etc.)

**Bad signs**:
- ❌ Many codes look identical (codebook collapse)
- ❌ Codes show random/nonsensical poses (poor reconstruction)
- ❌ Low perplexity during training (few codes used)

### Interpreting Reconstructions

**Good reconstruction**:
- Original (blue) and reconstructed (orange) poses overlap closely
- Key behavioral features preserved (direction, limb positions)
- Smooth trajectories (no jittering)

**Poor reconstruction**:
- Large gaps between original and reconstructed
- Missing limbs or distorted poses
- Discontinuous or jittery motion

### Common Patterns

1. **Stationary/Static codes**: Some codes may represent "do nothing" or neutral poses

2. **Directional codes**: Codes may specialize by direction (e.g., "turn left" vs "turn right")

3. **Speed codes**: Different codes for slow vs fast versions of same behavior

4. **Transition codes**: Some codes may represent transitions between behaviors

---

## How the Math Works

### From Codes to Behavior

Given discrete codes `[12, 42, 17, 5, 28]`:

```python
# Step 1: Codes → One-hot vectors (B, T, num_embeddings)
codes = torch.tensor([[12, 42, 17, 5, 28]])  # (1, 5)
one_hot = F.one_hot(codes, num_embeddings=64)  # (1, 5, 64)

# Step 2: One-hot × Codebook → Embeddings (B, T, embedding_dim)
codebook = model.quantizer.embedding.weight  # (64, 256)
embeddings = torch.matmul(one_hot.float(), codebook)  # (1, 5, 256)

# Step 3: Reshape for decoder (B, C, T)
embeddings = embeddings.permute(0, 2, 1)  # (1, 256, 5)

# Step 4: Decode → Behavior (B, 48, 150)
behavior = model.decoder(embeddings)  # (1, 48, 150)

# Step 5: Reshape for visualization (T, 24, 2)
behavior = behavior.reshape(24, 2, 150).permute(2, 0, 1)  # (150, 24, 2)
```

### Why Repeat the Same Code?

When visualizing a single codebook embedding (e.g., code 42):

```python
# Option 1: Repeat same code
codes = [42, 42, 42, 42, 42]
# Shows: Pure behavior pattern represented by code 42
# Like: "What is code 42 in isolation?"

# Option 2: Mix with other codes
codes = [12, 42, 17, 5, 28]
# Shows: How code 42 blends with other behaviors
# Like: "How does code 42 transition to/from other codes?"
```

**For pure codebook visualization**, we use **Option 1** (repeat same code)

---

## Advanced: Understanding Latent Length

### What is Latent Length?

- Input sequence: 150 frames
- After encoder downsampling: 5 latent timesteps
- **Latent length = 5**

### Why Does it Matter?

Each latent timestep covers ~30 frames (150 / 5 = 30):

```
Frame:   0────────30────────60────────90────────120───────150
Code:    [  12  ][  42  ][  17  ][   5  ][   28  ]
         └─30fr─┘└─30fr─┘└─30fr─┘└─30fr─┘└─30fr─┘
```

**Implication**: Each code represents ~30 frames of behavior

### Computing Latent Length

```python
# Method 1: From model architecture
latent_length = sequence_length // product(strides)
# Example: 150 // (5 × 3 × 2) = 5

# Method 2: From actual encoding
dummy = torch.zeros(1, 48, 150)
latent = model.encoder(dummy)
latent_length = latent.shape[-1]  # 5
```

---

## Troubleshooting

### "Checkpoint missing 'args' key"
- Your checkpoint doesn't contain training config
- Solution: Retrain and ensure training script saves `args` in checkpoint

### "All codes look the same"
- Codebook collapse - model not using full codebook
- Solutions:
  - Increase commitment_cost
  - Try improved VQ with dead code removal
  - Check reconstruction loss during training

### "Reconstructions are poor quality"
- Model underfitting or undertrained
- Solutions:
  - Train longer
  - Increase model capacity (hidden_dims)
  - Check if data preprocessing is correct

### "Out of memory"
- Decoding too many embeddings at once
- Solution: Use `--chunk_size 64` to decode in smaller batches

---

## Summary

### Visualization = Decode → Render

The core concept is simple:

1. **Start with**: Discrete code indices (integers)
2. **Lookup**: Embedding vectors from codebook
3. **Decode**: Pass through decoder CNN
4. **Render**: Convert to poses and visualize

### Three Scripts for Three Purposes

| Script | Purpose | Output |
|--------|---------|--------|
| `visualize_codebook_embeddings.py` | What does each code mean? | Pure behavior syllables |
| `visualize_reconstructions.py` | How well do we reconstruct? | Quality assessment |
| Custom codes (via `--codes`) | Can we compose behaviors? | Behavior generation |

### Key Takeaway

**The codebook is a learned vocabulary of behaviors.** Visualizing it helps you understand:
- What behaviors your model discovered
- Whether the discretization makes semantic sense
- How to interpret or generate behaviors from code sequences

---

## Next Steps

1. ✅ Run `visualize_codebook_embeddings.py` on your trained model
2. ✅ Examine the output images to understand what each code represents
3. ✅ Run `visualize_reconstructions.py` to assess quality
4. ✅ Experiment with custom code sequences to compose behaviors
5. ✅ Use these insights for downstream analysis (e.g., behavior clustering, transition analysis)

---

**Questions?** Check the inline comments in:
- `flies/vq_vae/vqvae.py` (model architecture)
- `flies/visualization/visualize_codebook_embeddings.py` (visualization script)
- `flies/visualization/reconstruction.py` (helper utilities)
