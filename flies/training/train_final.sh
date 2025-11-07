#!/bin/bash
# Training strategy based on actual data analysis
#
# KEY INSIGHT: All runs perform best at epoch 10-30, then degrade
# Root cause: LR decay too slow (T_max=100)
# Solution: Much faster LR decay

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
    --epochs 30 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --grad_clip_norm 1.0 \
    --lr_scheduler_tmax 20 \
    \
    --output_dir ./outputs/final_optimized \
    --save_every 5

# KEY CHANGES based on analysis:
# 1. epochs: 100 → 30 (all runs peaked before epoch 30)
# 2. lr_scheduler_tmax: 100 → 20 (decay LR much faster!)
# 3. save_every: 10 → 5 (catch the best model early)
# 4. num_embeddings: 32 → 64 (stable_training's 32 was too small)
# 5. Keep lr=1e-4 (works great initially, just needs fast decay)

# EXPECTED RESULTS:
# - Fast convergence in first 10 epochs (like all previous runs)
# - LR decays to ~0 by epoch 20 (prevents divergence)
# - Best model likely around epoch 10-15
# - Should match or beat: loss=81.86, recon=77.89, vq=3.97

# LOGIC:
# Your data shows:
# - Epoch 10: Great performance in all runs
# - Epoch 30+: Degradation in all runs
# → Stop training earlier! Don't waste compute on epochs 30-100
# → Use aggressive LR decay (T_max=20) to prevent divergence
