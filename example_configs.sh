#!/bin/bash
# Example commands for training with different quantization methods
#
# Make sure vector-quantize-pytorch is installed first:
#   pip install vector-quantize-pytorch

# Set your data paths
TRAIN_DATA="data/train.npy"  # Update this path
VAL_DATA="data/val.npy"      # Update this path
OUTPUT_DIR="outputs/quantizer_comparison"

# ============================================================================
# 1. QUICK TEST - Compare VQ, Improved VQ, and FSQ (20 epochs, ~1-2 hours)
# ============================================================================
echo "Running quick comparison test..."
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods vq vq_improved fsq \
    --epochs 20 \
    --lr_scheduler_tmax 15 \
    --output_dir $OUTPUT_DIR/quick_test \
    --save_every 5 \
    --grad_clip_norm 1.0

# ============================================================================
# 2. FULL COMPARISON - All methods (30 epochs, overnight)
# ============================================================================
echo "Running full comparison..."
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods vq vq_improved fsq rvq lfq \
    --epochs 30 \
    --lr_scheduler_tmax 20 \
    --output_dir $OUTPUT_DIR/full_comparison \
    --save_every 5 \
    --grad_clip_norm 1.0

# ============================================================================
# 3. INDIVIDUAL METHOD TRAINING (if you want to train just one)
# ============================================================================

# 3a. Standard VQ (baseline)
echo "Training baseline VQ..."
python flies/training/train.py \
    --model_type groupnorm \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --epochs 30 \
    --lr_scheduler_tmax 20 \
    --output_dir $OUTPUT_DIR/vq_baseline

# 3b. FSQ only (recommended first try)
echo "Training FSQ..."
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods fsq \
    --epochs 30 \
    --lr_scheduler_tmax 20 \
    --output_dir $OUTPUT_DIR/fsq_only

# 3c. Residual VQ only
echo "Training RVQ..."
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods rvq \
    --epochs 30 \
    --lr_scheduler_tmax 20 \
    --output_dir $OUTPUT_DIR/rvq_only

# ============================================================================
# 4. CUSTOM CONFIGURATIONS
# ============================================================================

# 4a. FSQ with different level configurations
# Try different level combinations to tune codebook size:

# Small codebook (~525 codes)
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods fsq \
    --epochs 30 \
    --output_dir $OUTPUT_DIR/fsq_small

# Large codebook (~2000 codes)
# Note: Need to modify QUANTIZER_CONFIGS in train_comparison.py to use [8,8,8,5]

# 4b. RVQ with different number of quantizers
# For 2 quantizers (faster training):
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods rvq \
    --epochs 30 \
    --output_dir $OUTPUT_DIR/rvq_2quantizers
# Note: Modify QUANTIZER_CONFIGS to set num_quantizers=2

# 4c. Different model sizes
# Smaller model (faster, less capacity)
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods fsq \
    --hidden_dims 32 64 128 \
    --embedding_dim 128 \
    --epochs 30 \
    --output_dir $OUTPUT_DIR/fsq_small_model

# Larger model (slower, more capacity)
python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --methods fsq \
    --hidden_dims 128 256 512 \
    --embedding_dim 512 \
    --epochs 30 \
    --output_dir $OUTPUT_DIR/fsq_large_model

# ============================================================================
# 5. WITH FLY-LEVEL SPLITS (if you have a split file)
# ============================================================================
FLY_SPLIT_FILE="data/fly_splits.json"  # Update this path

python flies/training/train_comparison.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --fly_split_file $FLY_SPLIT_FILE \
    --train_split_name train \
    --val_split_name val \
    --methods vq vq_improved fsq rvq \
    --epochs 30 \
    --output_dir $OUTPUT_DIR/with_fly_splits

# ============================================================================
# 6. ANALYSIS AFTER TRAINING
# ============================================================================

# After training completes, you can analyze results:
echo "Training complete! Check results in:"
echo "  - $OUTPUT_DIR/comparison_summary.json"
echo "  - $OUTPUT_DIR/{method_name}/training_history.json"
echo "  - $OUTPUT_DIR/{method_name}/best_model.pt"

# Each method will have its own directory with:
#   - best_model.pt: Best checkpoint (lowest val loss)
#   - final_model.pt: Final checkpoint
#   - training_history.json: Loss curves
#   - checkpoint_epoch_*.pt: Periodic checkpoints

# ============================================================================
# 6. ANALYSIS AFTER TRAINING - Compare all methods with plots!
# ============================================================================

echo ""
echo "Training complete! Check results in:"
echo "  - $OUTPUT_DIR/comparison_summary.json"
echo "  - $OUTPUT_DIR/{method_name}/training_history.json"
echo "  - $OUTPUT_DIR/{method_name}/best_model.pt"

# Quick look at summary
echo ""
echo "Quick summary:"
cat $OUTPUT_DIR/comparison_summary.json | python -m json.tool

echo ""
echo "============================================================================"
echo "Now run the analysis script to generate comparison plots:"
echo "============================================================================"
echo ""
echo "python flies/analysis/analyze_comparison.py --results_dir $OUTPUT_DIR"
echo ""

# Actually run the analysis (uncomment to auto-run after training)
# python flies/analysis/analyze_comparison.py --results_dir $OUTPUT_DIR

# This will generate:
#   - loss_curves.png: Training/validation loss for all methods
#   - vq_metrics.png: VQ loss and perplexity curves
#   - final_comparison_bars.png: Bar chart comparing final metrics
#   - learning_curves_grid.png: Individual learning curves per method
#   - summary_table.png: Visual summary table
#   - summary_table.txt: Text summary for easy viewing
