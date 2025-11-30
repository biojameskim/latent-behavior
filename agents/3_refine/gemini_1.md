Diagnosis: OPTION A: REFINE

The model has suffered from Codebook Collapse (only ~23% of codes used) and Training Instability (exploding loss). This renders the "discovered" behaviors scientifically invalid as they are likely just compression artifacts.

The Fix: I have replaced the standard 
VectorQuantizer
 in 
agents/2_code/gnn_vq_vae/pipeline.py
 with an EMA (Exponential Moving Average) Vector Quantizer.

Why?: EMA updates decouple the codebook learning from the optimizer, preventing "dead" codes and stabilizing training. This is the standard solution for collapse in VQ-VAEs.