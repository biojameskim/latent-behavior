Reality Check

Final ARI: 0.0
Final Reconstruction MSE: 0.9844
Post-Mortem

Sorting failure (collapse to one state): Codebook usage is 0.03125 and mean bout length is 300 (entire sequence), so the model converged to a single latent state. Despite stronger penalties, the decoder + encoder learned to satisfy reconstruction without needing multiple codes; the hinge penalties were still dominated by the incentive to keep codes constant given the simple semi-Markov prior and short sequences.

Root cause:

Trade-off imbalance: Reconstruction + commitment terms outweighed the entropy/usage and temporal-change penalties. The straight-through VQ combined with a simple GRU decoder made it easy to fit per-frame signals without switching codes.
Structural limitation: The generative prior is weak: transitions/durations are learned but not tied to data likelihood, and there is no posterior regularizer that forces distinct states to model distinct pose dynamics. Without contrastive or discriminative pressure, the VQ codes are unused “extra parameters” rather than necessary factors.
Glass ceiling / next major change:
Introduce a contrastive or predictive auxiliary objective that forces latents to discriminate between future segments (e.g., CPC/InfoNCE on segment embeddings) or a segment-level mutual information term (Barlow Twins/SimCLR on code-conditioned embeddings).
Alternatively, replace the VQ bottleneck with a true HSMM with emission tied to a state-conditioned flow or GMM and perform amortized inference over segments (e.g., structured VAE with semi-Markov posterior). This would make state usage necessary for likelihood, not optional.