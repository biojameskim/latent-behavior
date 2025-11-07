#!/bin/bash
# Consolidated training script with improved codebook utilization

# Key changes from your runs:
# 1. Reduce codebook size: 64 → 32 (match actual complexity)
# 2. Increase embedding_dim: 128 → 256 (give codes more expressiveness)
# 3. Keep commitment_cost at 0.25 (your original was best)
# 4. Add flag to choose model type

MODEL_TYPE=${1:-"groupnorm"}  # "groupnorm" or "simple"

if [ "$MODEL_TYPE" = "simple" ]; then
    TRAIN_SCRIPT="train_simple.py"
    OUTPUT_DIR="./outputs/simple_optimized"
    echo "Training with SIMPLE model (no GroupNorm)"
else
    TRAIN_SCRIPT="train.py"
    OUTPUT_DIR="./outputs/groupnorm_optimized"
    echo "Training with GroupNorm model"
fi

python $TRAIN_SCRIPT \
    --train_data ../../../../data/fly_data/fly_group_train.npy \
    --val_data ../../../../data/fly_data/fly_group_train.npy \
    --fly_split_file ../data/fly_data/fly_split.json \
    --train_split_name train \
    --val_split_name val \
    \
    --window_size 150 \
    --stride 75 \
    \
    --hidden_dims 64 128 256 \
    --embedding_dim 256 \
    --num_embeddings 32 \
    --num_residual_blocks 2 \
    --commitment_cost 0.25 \
    \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --beta1 0.9 \
    --beta2 0.99 \
    \
    --output_dir $OUTPUT_DIR \
    --save_every 10

# Usage:
#   ./train_optimized.sh             # Use GroupNorm (recommended)
#   ./train_optimized.sh simple      # Use simple version
#   ./train_optimized.sh groupnorm   # Use GroupNorm (explicit)

# Expected results with these settings:
# - VQ loss: < 5
# - Recon loss: < 80
# - Perplexity: > 20 / 32 codes (>60% utilization)
