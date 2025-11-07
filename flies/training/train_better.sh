#!/bin/bash
# Better training strategy: Use original lr with proper scheduling and early stopping

# Key insight: lr=1e-4 works great initially (epoch 1-10)
# Problem: It diverges later without LR decay
# Solution: Start with 1e-4 and decay it aggressively

python train_unified.py \
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
    --num_embeddings 64 \
    --num_residual_blocks 2 \
    --commitment_cost 0.25 \
    \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad_clip_norm 1.0 \
    --lr_scheduler_tmax 50 \
    \
    --output_dir ./outputs/better_training \
    --save_every 5

# Key changes from "stable" version:
# - lr: 5e-5 → 1e-4 (back to original - learns faster!)
# - num_embeddings: 32 → 64 (more codes for complexity)
# - epochs: 100 → 50 (shorter, rely on early stopping)
# - save_every: 10 → 5 (more checkpoints to catch best model)
# - lr_scheduler_tmax: 50 (CosineAnnealing decays LR over 50 epochs)

# Why this should work:
# 1. Fast learning initially (lr=1e-4) like original run
# 2. LR decays via cosine schedule (prevents divergence)
# 3. Gradient clipping prevents explosions
# 4. More frequent checkpoints to catch the best model
# 5. Val loss should help us stop at the right time

# Expected results:
# - Fast convergence in first 10 epochs (like original)
# - Stable training throughout (unlike original)
# - Best model likely around epoch 10-20
# - VQ loss < 5, Recon loss < 80
