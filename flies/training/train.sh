#!/bin/bash
# Fixed training script with stability improvements

# Key fixes for training instability:
# 1. Lower learning rate with warmup
# 2. Add gradient clipping
# 3. Use EMA for codebook updates (more stable)
# 4. Better hyperparameters based on analysis

python train.py \
    --model_type groupnorm \
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
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad_clip_norm 1.0 \
    \
    --output_dir ./outputs/stable_training \
    --save_every 10

# Changes from previous runs:
# - lr: 1e-4 → 5e-5 (lower to prevent divergence)
# - weight_decay: 0 → 1e-5 (add regularization)
# - grad_clip_norm: None → 1.0 (prevent gradient explosion)
# - num_embeddings: 64 → 32 (better utilization)
# - embedding_dim: 128 → 256 (more expressive)

# Expected results:
# - VQ loss should STAY < 10 throughout training (not increase to 40+!)
# - Perplexity should gradually increase
# - Training should be stable with no collapse
