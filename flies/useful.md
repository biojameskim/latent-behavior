# Useful details

Some useful commands I've been using:

## 1. Training the vanilla VQ-VAE model + some variations
I used `train_comparison.py` first to identify which method family works best, then used `train_rvq_fsq_comparison.py` to optimize RVQ/FSQ configurations since they were the best.

### General model training
```
python train_comparison.py \
--train_data ../../../../data/fly_data/fly_group_train.npy \
--val_data ../../../../data/fly_data/fly_group_train.npy \
--fly_split_file ../data/fly_data/fly_split.json \
\

--methods vq vq_improved fsq rvq lfq

\
--window_size 150 \
--stride 75 \
\
--embedding_dim 128 \
--num_embeddings 32 \
--num_residual_blocks 2 \
--commitment_cost 0.25 \
\
--batch_size 128 \
--epochs 20 \
--lr 1e-4 \
--weight_decay 1e-5 \
--lr_scheduler_tmax 15 \
\
--output_dir ./outputs/comparison \
--save_every 5
```
- **Purpose:** General comparison script for all 5 quantization methods
- **`--methods` available:** `vq`, `vq_improved`, `fsq`, `rvq`, `lfq`
- **Default hyperparameters:** More general (e.g., embedding_dim=256, lr=1e-4, batch_size=128, epochs=100)

### Fine-tuning between the RVQ/FSQ models 
- I was using this to compare different versions of rvq and fsq because they were the best out of all the VQ-VAE variations I trained. But I've also just been using this script to train the rvq_deep (the best one) which you can see I'm controlling using the `--methods` flag.
```
python train_rvq_fsq_comparison.py \
    --train_data ../../../../data/fly_data/fly_group_train.npy \
    --val_data ../../../../data/fly_data/fly_group_train.npy \
    --fly_split_file ../data/fly_data/fly_split.json \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --embedding_dim 128 \
    --num_embeddings 32 \
    --commitment_cost 0.25 \
    --output_dir outputs/rvq_fsq_comparison \
    --methods rvq_deep
```
- **Purpose:** Focused comparison of RVQ and FSQ variations only
- **`--methods` available:** `rvq_base`, `rvq_deep`, `rvq_large`, `rvq_xlarge`, `rvq_shared`, `rvq_shallow_large`, `fsq_base`, `fsq_small`, `fsq_large`, `fsq_balanced`, `fsq_minimal`, `fsq_xlarge`
- **Default hyperparameters:** Different defaults (e.g., embedding_dim=128, lr=1e-3, batch_size=32, epochs=20)

## 2. Generating figures for model training results
```
python analyze_comparison.py \
    --results_dir ../training/outputs/rvq_fsq_comparison
```

## 3. Visualizations
```
python visualize_reconstructions.py \
	--checkpoint ../training/outputs/rvq_fsq_comparison/rvq_deep/best_model.pt \
	--data_file ../../../../data/fly_data/fly_group_train.npy \
	--output_dir rvq_recon_viz_denorm \
	--window_overlays 10 \
	--window_frames 0 74 149 \
	--max_sequences 5 \
	--sequence_frames 0 500 1000 \
	--denormalize
```
- The `--denormalize` flag here is important. When creating the vq codebook, I want all the flies to start at (0,0) facing up so that they're normalized. But when reconstructing the visualizations, the multi-fly arena layout is meaningless because the spatial relationships between the flies is meaningless. So it's good to denormalize this for visualization. You can remove this flag, and you'll just get an egocentric view. Could be useful but don't try to interpret the spatial relationships between flies then.

We support **two types of visualizations**, each controlled by different parameters:

---

### **3a. Window Overlays (Close-up, per-fly snippets)**

Short, detailed clips (each = 150 frames) of **one fly** vs its reconstruction.

* **`window_overlays`**

  * *What it controls:* How many snippet windows to save.
  * *Example:* `window_overlays = 5` → save 5 different 150-frame windows.
  * *What you see:* Close-up reconstruction quality on individual behaviors.

* **`window_frames`**

  * *What it controls:* Which frames **within each 150-frame window** to visualize.
  * *Why needed:* Too many frames to save all 150.
  * *Example:* `window_frames = [0, 75, 149]` → show first, middle, last frame.
  * *What you see:* Snapshot comparisons (ground truth vs reconstruction) for each chosen frame.

**Together:**
`window_overlays = 5` and `window_frames = [0, 75, 149]` →
**5 windows × 3 frames each = 15 close-up comparison images**

---

### **3b. Sequence Overlays (Full-arena, all flies)**

Large-scale overview of the **entire 4500-frame trajectory**, showing **all flies** moving in the arena.

* **`max_sequences`**

  * *What it controls:* How many full-trajectory sequences to visualize.
  * *Example:* `max_sequences = 5` → show 5 different arenas.

* **`sequence_frames`**

  * *What it controls:* Which frame indices from each 4500-frame sequence to snapshot.
  * *Why needed:* Saving all 4500 frames is impossible/too heavy.
  * *Example:* `sequence_frames = [0, 1000, 2000]` → snapshot 3 points in time.
  * *What you see:* Arena-wide fly positions at chosen timestamps.

**Together:**
`max_sequences = 5` and `sequence_frames = [0, 1000, 2000]` →
**5 sequences × 3 frames each = 15 arena-wide trajectory images**

---

