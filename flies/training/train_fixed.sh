#!/bin/bash
# Training script with fixes for codebook collapse
# DO NOT use old checkpoints - must train from scratch!

# Recommended hyperparameters after fix:
# - embedding_dim increased from 32 to 128 (more capacity)
# - num_embeddings: 64-128 (moderate codebook size)
# - GroupNorm will normalize encoder outputs before quantization
# - Codebook initialized with normal(0,1) instead of tiny uniform range

python train.py \
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
    --embedding_dim 128 \
    --num_embeddings 64 \
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
    --output_dir ./outputs/fixed_normalization_run \
    --save_every 10

# Expected results after fix:
# - VQ loss should be < 10 (was 640!)
# - Perplexity should be > 50 / 64 codes
# - Total loss should be < 100 after a few epochs

# Monitor with:
# tail -f outputs/fixed_normalization_run/training.log
