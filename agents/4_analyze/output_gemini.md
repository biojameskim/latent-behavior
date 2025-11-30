Post-Mortem Analysis: GNN-VQ-VAE
The Reality Check
Final ARI: 0.0154
Final Reconstruction MSE: 0.2843
1. The "Sorting" Failure
The Paradox: We achieved excellent reconstruction (MSE 0.28, significantly better than the initial 0.94) and near-perfect codebook utilization (98.4%). Yet, the ARI is effectively zero (0.015).

Why? The "Compression vs. Semantics" Trap. The model successfully solved the optimization problem we gave it: minimize reconstruction error using a discrete bottleneck. It did this by becoming a highly efficient compression algorithm, not a behavior discoverer.

The high codebook usage (98%) and short bout length (~5.7 frames, or ~190ms) suggest the model learned "micro-movements" or "phonemes" of motion (e.g., "leg lift," "wing flick") rather than high-level "words" or behaviors (e.g., "grooming," "walking").
It chopped the continuous stream into tiny, reusable primitives that reconstruct well but have no one-to-one mapping with human labels. A single human label like "grooming" is likely composed of a complex sequence of these micro-codes (A-B-A-C-A), while "walking" is (D-E-F-D). The VQ-VAE sees the alphabet, but the human labels are the sentences.
2. Root Cause Analysis
Structural Limitation: The Lack of Temporal Hierarchy. The failure wasn't a trade-off; it was a mismatch in abstraction level.

No Temporal Aggregation: The VQ-VAE quantizes every frame (or small window). It has no mechanism to enforce that a "code" must correspond to a long-term state. The "temporal coherence" loss was a band-aid that penalized switching, but it didn't structure the latent space into long-term modes.
Euclidean Distance Fallacy: The VQ step uses Euclidean distance in latent space. However, behavior is defined by dynamics (how things change), not just statics (how things look). Two frames of "grooming" might look very different (leg up vs. leg down), but they belong to the same behavior. The VQ-VAE treats them as different codes because they are far apart in Euclidean space.
3. The "Glass Ceiling"
To break through this plateau, we need to move beyond frame-level reconstruction.

The Next Step: Self-Supervised Contrastive Learning (e.g., TCN or SimCLR for Time Series).

Why? Instead of asking "Can you reconstruct this pixel?", we should ask "Does this window of time belong to the same underlying state as that window?"
Mechanism: Train an encoder to pull together different crops/views of the same temporal window and push away distant windows. This forces the model to learn slowly varying features (semantics) rather than fast varying details (pose).
Hierarchical VQ-VAE: Alternatively, stack another VQ-VAE on top of the learned codes to find "patterns of patterns" (words formed from letters). This would bridge the gap between the micro-codes we found and the macro-labels we want.