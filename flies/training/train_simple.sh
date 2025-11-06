#!/bin/bash
# Training script with MINIMAL fix (initialization only, no normalization)
# This is closer to the original VQ-VAE paper

# Try this FIRST! If it works, it's the simplest solution.
# If you still get high VQ loss (>50), use train_fixed.sh instead.

python train_simple.py \
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
    --output_dir ./outputs/simple_init_fix_run \
    --save_every 10

# Expected results:
# - VQ loss should be < 50 (ideally < 20)
# - If VQ loss is still > 50, use the GroupNorm version instead (train_fixed.sh)
#
# This version is simpler and closer to the original VQ-VAE paper.
