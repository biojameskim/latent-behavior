**STEP 4: Post-Mortem Analysis**

We are stopping the optimization loop here.
Here are the logs from the final run.

**The Reality Check:**
*   **Status:** CRASHED during evaluation.
*   **Error:** `DiscreteStateDecoder.forward() takes 2 positional arguments but 3 were given` (in [generate()](cci:1://file:///share/j_sun/jjk297/repos/latent-behavior/agents/2_code/gnn_vq_vae/pipeline.py:343:4-364:36)).
*   **Performance before crash:** Reconstruction was poor (0.90) and the model was stuck in "DEGENERATE_SEGMENTATION" (temporal change ~0.0).

**Task 1: The Post-Mortem**
Act as the Principal Investigator.
1.  **The Crash:** Briefly explain the bug in [generate()](cci:1://file:///share/j_sun/jjk297/repos/latent-behavior/agents/2_code/gnn_vq_vae/pipeline.py:343:4-364:36) that caused the crash.
2.  **The "Freezing" Failure:** Even before the crash, the model refused to switch states (Temporal Change = 0.0000). Why did the HSLDS fail to learn transitions?
3.  **Root Cause Analysis:** Why did this architecture (switching linear dynamics) prove so difficult to train compared to a standard VQ-VAE?
4.  **The "Glass Ceiling":** If we had infinite time, what would be the *next* major architectural change needed to break through this plateau?

**Output:**
Provide a brutally honest Post-Mortem analysis. Do not write any more code.

**FINAL LOGS START**
==========================================
HSLDS Behavior Discovery Pipeline
==========================================
Job started at: Sat Nov 29 09:42:31 PM EST 2025
Running on node: desa-compute-01.cs.cornell.edu
Job ID: 548522
Working directory: /share/j_sun/jjk297/repos/latent-behavior
Python version: Python 3.14.0
Which Python: /home/jjk297/.conda/envs/lat-beh/bin/python
Conda environment: lat-beh
Conda prefix: /home/jjk297/.conda/envs/lat-beh

==========================================
GPU Information:
==========================================
Sat Nov 29 21:42:31 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:B1:00.0 Off |                  N/A |
| 27%   24C    P8              9W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

==========================================
Checking dependencies...
==========================================
Dependencies checked/installed

==========================================
Running installation test...
==========================================
Testing installation...
✓ All imports successful
✓ Model instantiated (361,569 parameters)
✓ Preprocessing works (output shape: torch.Size([2, 50, 121]))
✓ Forward pass works (recon: torch.Size([2, 50, 121]), codes: torch.Size([2, 50]))
✗ Forward pass error: DiscreteStateDecoder.forward() takes 2 positional arguments but 3 were given

==========================================
Starting HSLDS Training...
==========================================
Data path: /share/j_sun/jjk297/data/fly_data/mabe22_subset_for_claude.npz
Device: cuda
Epochs: 100
Batch size: 32
Number of states: 6
Latent dimension: 32

Using device: cuda

============================================================
LOADING DATA
============================================================
Loaded dataset:
  Trajectories shape: (50, 300, 48)
  Labels shape: (50, 300)
  Keypoints: 24
  Number of unique behaviors: 3

============================================================
INITIALIZING MODEL
============================================================
Architecture: Hierarchical Switching Linear Dynamical System
  Input dimension:  48
  Latent dimension: 32
  Number of states: 6
  Total parameters: 359,253

============================================================
TRAINING
============================================================
Training on 50 sequences for 100 epochs
Device: cuda

[WARNING] Failure modes detected: ['TEMPORAL_FLICKERING', 'POOR_RECONSTRUCTION']
Epoch 1/100 - Loss: 1.8744, Recon: 0.9588, Codebook: 0.4640, Temporal: 0.6142

[WARNING] Failure modes detected: ['TEMPORAL_FLICKERING', 'POOR_RECONSTRUCTION']
Epoch 2/100 - Loss: 1.8508, Recon: 0.9776, Codebook: 0.5033, Temporal: 0.5889

[WARNING] Failure modes detected: ['TEMPORAL_FLICKERING', 'POOR_RECONSTRUCTION']
Epoch 3/100 - Loss: 1.7890, Recon: 0.9460, Codebook: 0.5451, Temporal: 0.5748

[WARNING] Failure modes detected: ['TEMPORAL_FLICKERING', 'POOR_RECONSTRUCTION']
Epoch 4/100 - Loss: 1.8024, Recon: 1.0084, Codebook: 0.6011, Temporal: 0.5415

[WARNING] Failure modes detected: ['POOR_RECONSTRUCTION']
Epoch 5/100 - Loss: 1.6824, Recon: 0.9230, Codebook: 0.6388, Temporal: 0.5241

[WARNING] Failure modes detected: ['POOR_RECONSTRUCTION']
Epoch 6/100 - Loss: 1.7048, Recon: 1.0025, Codebook: 0.7065, Temporal: 0.4857

[WARNING] Failure modes detected: ['POOR_RECONSTRUCTION']
Epoch 7/100 - Loss: 1.5842, Recon: 0.9421, Codebook: 0.7881, Temporal: 0.4469

[WARNING] Failure modes detected: ['POOR_RECONSTRUCTION']
Epoch 8/100 - Loss: 1.5034, Recon: 0.9409, Codebook: 0.8861, Temporal: 0.3921

[WARNING] Failure modes detected: ['POOR_RECONSTRUCTION']
Epoch 9/100 - Loss: 1.4722, Recon: 1.0029, Codebook: 1.0101, Temporal: 0.3278

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 10/100 - Loss: 1.3354, Recon: 0.9907, Codebook: 1.2132, Temporal: 0.2338

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 11/100 - Loss: 1.1699, Recon: 0.9379, Codebook: 1.3936, Temporal: 0.1492

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 12/100 - Loss: 1.0747, Recon: 0.9404, Codebook: 1.5776, Temporal: 0.0715

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 13/100 - Loss: 1.0604, Recon: 0.9814, Codebook: 1.6978, Temporal: 0.0267

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 14/100 - Loss: 1.0461, Recon: 0.9869, Codebook: 1.7458, Temporal: 0.0115

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 15/100 - Loss: 0.9862, Recon: 0.9360, Codebook: 1.7703, Temporal: 0.0046

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 16/100 - Loss: 1.0111, Recon: 0.9653, Codebook: 1.7847, Temporal: 0.0015

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 17/100 - Loss: 1.0174, Recon: 0.9725, Codebook: 1.7850, Temporal: 0.0014

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 18/100 - Loss: 0.9475, Recon: 0.9042, Codebook: 1.7883, Temporal: 0.0007

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 19/100 - Loss: 1.0325, Recon: 0.9901, Codebook: 1.7894, Temporal: 0.0004

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 20/100 - Loss: 1.0057, Recon: 0.9637, Codebook: 1.7887, Temporal: 0.0006

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 21/100 - Loss: 0.9873, Recon: 0.9464, Codebook: 1.7903, Temporal: 0.0002

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 22/100 - Loss: 0.9525, Recon: 0.9123, Codebook: 1.7909, Temporal: 0.0002

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 23/100 - Loss: 0.9936, Recon: 0.9539, Codebook: 1.7895, Temporal: 0.0003

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 24/100 - Loss: 1.0448, Recon: 1.0060, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 25/100 - Loss: 1.0041, Recon: 0.9659, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 26/100 - Loss: 0.9840, Recon: 0.9465, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 27/100 - Loss: 0.9928, Recon: 0.9559, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 28/100 - Loss: 0.9697, Recon: 0.9334, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 29/100 - Loss: 0.9628, Recon: 0.9270, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 30/100 - Loss: 0.9977, Recon: 0.9626, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 31/100 - Loss: 0.9775, Recon: 0.9430, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 32/100 - Loss: 1.0288, Recon: 0.9949, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 33/100 - Loss: 0.9453, Recon: 0.9120, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 34/100 - Loss: 0.9533, Recon: 0.9205, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 35/100 - Loss: 0.9348, Recon: 0.9027, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 36/100 - Loss: 0.9832, Recon: 0.9515, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 37/100 - Loss: 0.9947, Recon: 0.9637, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 38/100 - Loss: 0.9861, Recon: 0.9555, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 39/100 - Loss: 0.9853, Recon: 0.9552, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 40/100 - Loss: 0.9669, Recon: 0.9374, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 41/100 - Loss: 0.9480, Recon: 0.9191, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 42/100 - Loss: 1.0170, Recon: 0.9886, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 43/100 - Loss: 0.9708, Recon: 0.9427, Codebook: 1.7909, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 44/100 - Loss: 0.9298, Recon: 0.9022, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 45/100 - Loss: 0.9561, Recon: 0.9289, Codebook: 1.7909, Temporal: 0.0002

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 46/100 - Loss: 0.9789, Recon: 0.9523, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 47/100 - Loss: 1.0258, Recon: 0.9997, Codebook: 1.7909, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 48/100 - Loss: 0.9529, Recon: 0.9273, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 49/100 - Loss: 0.9826, Recon: 0.9575, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 50/100 - Loss: 0.9207, Recon: 0.8960, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 51/100 - Loss: 0.9994, Recon: 0.9751, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 52/100 - Loss: 0.9570, Recon: 0.9331, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 53/100 - Loss: 0.9577, Recon: 0.9342, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 54/100 - Loss: 0.9460, Recon: 0.9228, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 55/100 - Loss: 0.9397, Recon: 0.9169, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 56/100 - Loss: 0.9492, Recon: 0.9268, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 57/100 - Loss: 0.9748, Recon: 0.9526, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 58/100 - Loss: 0.9776, Recon: 0.9558, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 59/100 - Loss: 0.9408, Recon: 0.9193, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 60/100 - Loss: 0.9555, Recon: 0.9344, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 61/100 - Loss: 0.9424, Recon: 0.9217, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 62/100 - Loss: 0.9711, Recon: 0.9505, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 63/100 - Loss: 1.0030, Recon: 0.9827, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 64/100 - Loss: 0.9203, Recon: 0.9002, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 65/100 - Loss: 0.9272, Recon: 0.9073, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 66/100 - Loss: 0.9825, Recon: 0.9627, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 67/100 - Loss: 0.9918, Recon: 0.9723, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 68/100 - Loss: 0.9399, Recon: 0.9206, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 69/100 - Loss: 0.9394, Recon: 0.9202, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 70/100 - Loss: 0.9802, Recon: 0.9612, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 71/100 - Loss: 0.9712, Recon: 0.9523, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 72/100 - Loss: 0.9501, Recon: 0.9314, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 73/100 - Loss: 0.9919, Recon: 0.9733, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 74/100 - Loss: 0.9276, Recon: 0.9092, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 75/100 - Loss: 0.9500, Recon: 0.9317, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 76/100 - Loss: 0.9788, Recon: 0.9606, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 77/100 - Loss: 0.9992, Recon: 0.9811, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 78/100 - Loss: 0.9425, Recon: 0.9244, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 79/100 - Loss: 0.9550, Recon: 0.9369, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 80/100 - Loss: 0.9621, Recon: 0.9439, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 81/100 - Loss: 0.9569, Recon: 0.9387, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 82/100 - Loss: 0.9553, Recon: 0.9371, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 83/100 - Loss: 0.9414, Recon: 0.9233, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 84/100 - Loss: 0.9473, Recon: 0.9292, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 85/100 - Loss: 0.9448, Recon: 0.9267, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 86/100 - Loss: 0.9078, Recon: 0.8898, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 87/100 - Loss: 0.9586, Recon: 0.9406, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 88/100 - Loss: 0.9872, Recon: 0.9692, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 89/100 - Loss: 0.9557, Recon: 0.9377, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 90/100 - Loss: 0.9748, Recon: 0.9568, Codebook: 1.7912, Temporal: 0.0001

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 91/100 - Loss: 0.9848, Recon: 0.9668, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 92/100 - Loss: 0.9195, Recon: 0.9015, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 93/100 - Loss: 0.9076, Recon: 0.8896, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 94/100 - Loss: 0.9657, Recon: 0.9477, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 95/100 - Loss: 0.9768, Recon: 0.9588, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 96/100 - Loss: 0.9286, Recon: 0.9106, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 97/100 - Loss: 0.9976, Recon: 0.9796, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 98/100 - Loss: 0.9299, Recon: 0.9119, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 99/100 - Loss: 0.9763, Recon: 0.9583, Codebook: 1.7918, Temporal: 0.0000

[WARNING] Failure modes detected: ['DEGENERATE_SEGMENTATION', 'POOR_RECONSTRUCTION']
Epoch 100/100 - Loss: 0.9217, Recon: 0.9037, Codebook: 1.7918, Temporal: 0.0000
Training history saved to output/hslds_20251129_214239/training_history.png

============================================================
INTRINSIC EVALUATION (No Ground Truth)
============================================================
Running intrinsic evaluation...
  - Computing reconstruction MSE...
  - Analyzing code usage...
  - Computing temporal statistics...
  - Generating synthetic data...

ERROR: Training failed!

**FINAL LOGS END**