# RVQ (Residual Vector Quantization) Visualization Guide

## What is RVQ?

**RVQ** extends standard VQ by using **multiple quantizers in a hierarchical residual structure**.

### Standard VQ vs RVQ

```
Standard VQ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input → Encoder → Quantizer → Decoder
                     ↓
                  One code per timestep
                  [12, 42, 17, 5, 28]
                  Shape: (batch, time)

RVQ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input → Encoder → Q0 → Q1 → Q2 → Q3 → Decoder
                   ↓    ↓    ↓    ↓
               Multiple codes per timestep (summed)
               [[12,5,3,1], [42,17,9,2], ...]
               Shape: (batch, time, num_quantizers)
```

### How RVQ Works

RVQ uses **residual quantization**:

1. **First quantizer (Q0)**: Quantizes the input
   ```
   z → z_q0 (quantized)
   residual_1 = z - z_q0
   ```

2. **Second quantizer (Q1)**: Quantizes the residual
   ```
   residual_1 → z_q1 (quantized)
   residual_2 = residual_1 - z_q1
   ```

3. **Continue** for num_quantizers stages

4. **Final output**: Sum all quantized vectors
   ```
   output = z_q0 + z_q1 + z_q2 + z_q3
   ```

### Key Differences

| Aspect | Standard VQ | RVQ |
|--------|-------------|-----|
| **Codes per timestep** | 1 | num_quantizers (e.g., 4) |
| **Index shape** | (batch, time) | (batch, time, num_quantizers) |
| **Effective codebook** | K codes | K^Q codes (exponential!) |
| **Capacity** | Limited by K | Much higher |
| **Hierarchy** | Flat | Hierarchical (coarse → fine) |

### Example

Standard VQ with codebook size 64:
- Total unique behaviors: **64**
- One code per timestep: `[12, 42, 17, 5, 28]`

RVQ with codebook size 64, 4 quantizers:
- Total unique behaviors: **64^4 = 16,777,216** (!)
- Four codes per timestep: `[[12,5,3,1], [42,17,9,2], [17,28,15,7], ...]`

This is why RVQ is much more powerful!

---

## Visualizing RVQ Codebooks

### Challenge

With RVQ, "visualizing the codebook" is ambiguous because:
- Each timestep has **multiple codes**
- Codes combine additively (hierarchical)
- The space is exponentially large

### Three Visualization Approaches

We provide three specialized visualization modes:

#### 1. Individual Quantizers

**Purpose**: Understand what each quantizer learns independently

**Method**: Visualize codes from ONE quantizer, setting others to 0

```python
# Visualize code 42 from quantizer 0
indices = [[42, 0, 0, 0],    # Timestep 0: only Q0=42
           [42, 0, 0, 0],    # Timestep 1: only Q0=42
           ...]

# Visualize code 17 from quantizer 2
indices = [[0, 0, 17, 0],    # Only Q2=17
           [0, 0, 17, 0],
           ...]
```

**Reveals**:
- What does each quantizer specialize in?
- Typically: Q0 = coarse structure, Q1-Q3 = fine details
- Example findings:
  - Q0: Overall body position and gross movement
  - Q1: Limb orientations
  - Q2: Fine leg positions
  - Q3: Subtle corrections

#### 2. Full Combinations

**Purpose**: See what different code combinations produce

**Method**: Sample random combinations of all quantizers

```python
# Random combination
indices = [[12, 5, 28, 3],   # All quantizers active
           [42, 17, 9, 1],
           [17, 28, 15, 7],
           ...]
```

**Reveals**:
- How do codes combine?
- The diversity of the learned space
- Natural vs unnatural combinations

#### 3. Ablation Studies

**Purpose**: Understand hierarchical structure by progressively removing quantizers

**Method**: Start with full codes, then zero out higher quantizers

```python
# Full
[[12, 5, 28, 3], [42, 17, 9, 1], ...]

# Without Q3
[[12, 5, 28, 0], [42, 17, 9, 0], ...]

# Without Q2 & Q3
[[12, 5, 0, 0], [42, 17, 0, 0], ...]

# Only Q0
[[12, 0, 0, 0], [42, 0, 0, 0], ...]
```

**Reveals**:
- Progressive refinement from coarse to fine
- How much each quantizer contributes
- Whether higher quantizers add meaningful detail

---

## Usage

### Option 1: Use RVQ-Specific Script (Recommended)

For detailed hierarchical visualizations:

```bash
python visualize_rvq_codebook.py \
    --checkpoint ../training/outputs/rvq_run/best_model.pt \
    --output_dir rvq_viz \
    --mode all \
    --num_samples 10
```

**Modes**:
- `individual`: Visualize each quantizer's codes independently
- `combinations`: Random combinations of all quantizers
- `ablations`: Progressive ablation study
- `all`: All of the above

**Output**:
```
rvq_viz/
├── individual_quantizers/
│   ├── quantizer_0_code_0000.png
│   ├── quantizer_0_code_0010.png
│   ├── quantizer_1_code_0000.png
│   └── ...
├── combinations/
│   ├── combination_000.png
│   ├── combination_001.png
│   └── ...
└── ablations/
    ├── ablation_000.png  # Shows full → Q0-Q2 → Q0-Q1 → Q0 only
    ├── ablation_001.png
    └── ...
```

### Option 2: Use Standard Script (Quick)

The standard script also works with RVQ (auto-detected):

```bash
python visualize_codebook_embeddings.py \
    --checkpoint ../training/outputs/rvq_run/best_model.pt \
    --output_dir codebook_viz
```

**Note**: This visualizes codes only in the **first quantizer** (others set to 0).
- Good for: Quick overview of coarse behaviors
- Limitation: Doesn't show hierarchical structure

---

## Interpreting RVQ Visualizations

### Individual Quantizers

**What to look for**:

✅ **Good hierarchy**:
- Q0: Coarse, distinct behaviors (walking, turning, resting)
- Q1: Adds directional/orientational detail
- Q2: Adds fine limb positions
- Q3: Subtle corrections

❌ **Poor hierarchy**:
- All quantizers look similar (not specialized)
- Higher quantizers show random noise (not learning)
- Q0 alone produces perfect reconstructions (higher quantizers unused)

### Combinations

**What to look for**:

✅ **Good combinations**:
- Diverse behaviors from different code combinations
- Natural, smooth poses
- Clear behavioral interpretations

❌ **Poor combinations**:
- Many combinations look identical (codebook collapse)
- Unnatural or distorted poses
- No clear structure

### Ablations

**What to look for**:

✅ **Good progressive refinement**:
```
Q0 only:        Rough pose, correct general behavior
Q0 + Q1:        Better limb positions
Q0 + Q1 + Q2:   Fine details emerge
Full (Q0-Q3):   Subtle final corrections
```

Each level adds meaningful detail.

❌ **Poor hierarchy**:
- No visible difference between levels (quantizers redundant)
- Huge jump from Q0 to Q0+Q1 (hierarchy too steep)
- Higher levels make things worse (artifacts)

---

## Why Repeat Codes in RVQ?

Same principle as standard VQ, but applied hierarchically:

```python
# Repeat [42, 5, 3, 1] across all timesteps
indices = [[42, 5, 3, 1],
           [42, 5, 3, 1],
           [42, 5, 3, 1],
           [42, 5, 3, 1],
           [42, 5, 3, 1]]
```

This shows the **sustained behavior** for this particular code combination.

Remember:
- Each code is at a different latent timestep
- Decoder upsamples each position to ~30 frames
- The combination encodes a **motion pattern**, not a static pose

See `UNDERSTANDING_REPEATED_CODES.md` for detailed explanation.

---

## Understanding the Index Format

### Standard VQ
```python
indices.shape = (batch, time)
# Example: (1, 5) for one sequence with 5 latent timesteps
indices = [[12, 42, 17, 5, 28]]
```

### RVQ
```python
indices.shape = (batch, time, num_quantizers)
# Example: (1, 5, 4) for one sequence with 5 timesteps, 4 quantizers
indices = [[[12, 5, 3, 1],     # Timestep 0: codes from Q0, Q1, Q2, Q3
            [42, 17, 9, 2],    # Timestep 1
            [17, 28, 15, 7],   # Timestep 2
            [5, 32, 21, 11],   # Timestep 3
            [28, 8, 19, 4]]]   # Timestep 4
```

---

## Practical Tips

### 1. Start with Individual Quantizers

Understand what each level of the hierarchy learns before looking at combinations.

### 2. Use Ablations to Verify Hierarchy

Check that each quantizer adds meaningful detail, not just noise.

### 3. Compare to Standard VQ

Train both standard VQ and RVQ to see if the extra capacity helps:
- Does RVQ reconstruct better?
- Does RVQ capture more behaviors?
- Is the hierarchy interpretable?

### 4. Visualize Real Reconstructions

Use `visualize_reconstructions.py` (works with RVQ!) to see quality on real data:

```bash
python visualize_reconstructions.py \
    --checkpoint ../training/outputs/rvq_run/best_model.pt \
    --data_file ../../data/fly_data/fly_group_train.npy \
    --output_dir rvq_recon_viz
```

---

## Common Issues

### "All quantizers look the same"

**Problem**: Quantizers not specializing

**Solutions**:
- Reduce shared_codebook (use False for independent codebooks)
- Increase model capacity
- Train longer
- Check if lower quantizers have sufficient capacity

### "Higher quantizers add noise, not detail"

**Problem**: Hierarchy collapsing or overfitting

**Solutions**:
- Reduce commitment_cost for higher quantizers
- Add dropout or regularization
- Reduce num_quantizers (maybe 2-3 is enough)
- Check reconstruction loss curve

### "Reconstruction no better than standard VQ"

**Problem**: Not utilizing extra capacity

**Solutions**:
- Ensure you're giving model enough training data
- Try larger codebook_size per quantizer
- Verify training is working (check perplexity, loss curves)

---

## Summary

### Key Takeaways

1. **RVQ = Hierarchical quantization**: Multiple codes per timestep, summed residually

2. **Much larger effective codebook**: K^Q instead of K codes

3. **Three visualization modes**:
   - Individual: What does each quantizer learn?
   - Combinations: How do codes combine?
   - Ablations: How does the hierarchy refine?

4. **Use specialized script**: `visualize_rvq_codebook.py` for full analysis

5. **Check hierarchy**: Good RVQ shows coarse → fine progression

### When to Use RVQ

✅ Use RVQ when:
- Standard VQ reconstruction quality is insufficient
- You have complex, high-dimensional behavior
- You want hierarchical structure (interpretable levels)

❌ Stick with standard VQ when:
- Reconstruction is already good
- Computational resources are limited
- You want simpler, more interpretable codes

---

## Further Reading

- `CODEBOOK_VISUALIZATION_GUIDE.md` - Core visualization concepts
- `UNDERSTANDING_REPEATED_CODES.md` - Why repeated codes show motion
- [RVQ Paper: "SoundStream" (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312)
- [vector-quantize-pytorch library docs](https://github.com/lucidrains/vector-quantize-pytorch)
