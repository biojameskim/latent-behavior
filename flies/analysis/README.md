# Analysis Tools for VQ-VAE Comparison

## Quick Start

After training with `train_comparison.py`, analyze and visualize results:

```bash
python flies/analysis/analyze_comparison.py --results_dir outputs/group_norm_v5
```

## What It Does

The script automatically:
1. ✅ Loads training history from all methods
2. ✅ Creates comprehensive comparison plots
3. ✅ Generates summary tables
4. ✅ Saves everything to `outputs/{your_dir}/analysis_plots/`

## Generated Plots

### 1. **loss_curves.png** - Overall Loss Comparison
- Training and validation loss curves
- Reconstruction loss curves
- All methods overlaid for easy comparison

### 2. **vq_metrics.png** - Quantization Quality
- VQ loss over time (how well quantization works)
- Perplexity over time (codebook utilization)
- Higher perplexity = better codebook usage

### 3. **final_comparison_bars.png** - Final Performance
- Bar charts comparing final epoch metrics
- Highlights best method with gold border
- Shows validation loss, reconstruction, VQ loss, perplexity

### 4. **learning_curves_grid.png** - Individual Method Details
- Separate plots for each method
- Loss components broken down (total, reconstruction, VQ)
- Easy to spot training issues per method

### 5. **summary_table.png** - Quick Overview
- Table with all key metrics
- Final and best values
- Number of epochs trained

### 6. **summary_table.txt** - Text Summary
- Plain text version for easy viewing
- Copy-paste friendly format

## Usage Examples

### Analyze All Methods
```bash
# Auto-detects all trained methods
python flies/analysis/analyze_comparison.py \
    --results_dir outputs/group_norm_v5
```

### Analyze Specific Methods Only
```bash
# Only compare VQ, FSQ, and RVQ
python flies/analysis/analyze_comparison.py \
    --results_dir outputs/group_norm_v5 \
    --methods vq fsq rvq
```

## Expected Directory Structure

```
outputs/group_norm_v5/
├── vq/
│   ├── training_history.json
│   ├── best_model.pt
│   └── final_model.pt
├── vq_improved/
│   ├── training_history.json
│   └── ...
├── fsq/
│   ├── training_history.json
│   └── ...
├── rvq/
│   └── ...
├── lfq/
│   └── ...
├── comparison_summary.json
└── analysis_plots/  ← Created by this script
    ├── loss_curves.png
    ├── vq_metrics.png
    ├── final_comparison_bars.png
    ├── learning_curves_grid.png
    ├── summary_table.png
    └── summary_table.txt
```

## Interpreting Results

### Which Method Won?

Look at `final_comparison_bars.png`:
- **Lowest validation loss** = best overall
- **Lowest reconstruction loss** = best at reconstructing trajectories
- **Lowest VQ loss** = encoder and codebook are well-aligned
- **Highest perplexity** = using more of the codebook (good!)

### Is Training Stable?

Look at `loss_curves.png`:
- ✅ **Smooth decreasing curve** = stable training
- ❌ **Increasing after epoch N** = overfitting or LR schedule issue
- ❌ **Spiky/noisy** = may need gradient clipping or lower LR

### Codebook Collapse?

Look at `vq_metrics.png` (Perplexity plot):
- **Perplexity ≈ codebook_size** = all codes being used (perfect!)
- **Perplexity < 10** = codebook collapse (bad!)
- **FSQ doesn't have perplexity** = it's codebook-free

### Example Interpretation

```
Method       Val Loss  Recon Loss  VQ Loss  Perplexity
VQ           82.5      78.2        4.3      5.8        ← Baseline (collapse!)
VQ_IMPROVED  73.1      69.5        3.6      14.2       ← Better utilization
FSQ          68.4      68.4        N/A      N/A        ← Best reconstruction
RVQ          65.2      62.1        3.1      25.6       ← WINNER! Best overall
LFQ          71.8      67.2        4.6      18.3       ← Good, needs tuning

Conclusion: RVQ is best - lowest losses, highest codebook usage
```

## Tips

### Training Not Finished?
The script works with partial results! It will plot whatever epochs have completed.

### Compare Different Training Runs?
Just point to different directories:
```bash
python flies/analysis/analyze_comparison.py --results_dir outputs/run1
python flies/analysis/analyze_comparison.py --results_dir outputs/run2
```

### Want to Customize Plots?
Edit `analyze_comparison.py` - all plotting functions are clearly labeled.

## Dependencies

Required packages (should already be installed):
- matplotlib
- seaborn
- numpy
- json (standard library)

If missing:
```bash
pip install matplotlib seaborn numpy
```
