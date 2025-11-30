Excellent selection. I approve your primary architecture choice. Now, you are a Senior ML Engineer.

Phase 1: Implementation (Universal Pipeline)

Implement the complete pipeline for your single selected architecture. Place all artifacts in the `agents/2_code/<architecture>` directory. 

Universal Interface Requirement:

Regardless of the architecture you chose (VQ-VAE, HMM, Transformer), your class MUST adhere to the DiscoveryPipeline interface defined below. This ensures we can use the exact same evaluation metrics to compare different model types.

Required Components:

1. Preprocessing Module

Python



class BehaviorPreprocessor:

    """

    Critical preprocessing for behavior invariance

    """

    def __init__(self):

        pass

    

    def ego_centric_alignment(self, keypoints):

        """Center on reference point (e.g., thorax)"""

        # YOUR CODE: Convert absolute to relative coordinates

        pass

    

    def compute_velocities(self, keypoints):

        """Add temporal derivatives"""

        # YOUR CODE: velocity, acceleration

        pass

    

    def engineer_features(self, keypoints):

        """Domain-specific features"""

        # YOUR CODE: inter-keypoint distances, angles, etc.

        pass

    

    def normalize(self, features):

        """Scale normalization"""

        # YOUR CODE: z-score or min-max

        pass

    

    def preprocess(self, raw_trajectories):

        """Full pipeline"""

        aligned = self.ego_centric_alignment(raw_trajectories)

        velocities = self.compute_velocities(aligned)

        features = self.engineer_features(aligned)

        normalized = self.normalize(features)

        return normalized

2. Universal Model Architecture

Python



class DiscoveryPipeline(nn.Module):

    """

    Standardized wrapper for ANY generative behavior model.

    """

    

    def __init__(self, ...):

        super().__init__()

        # Initialize your specific architecture (VQ-VAE, HMM, etc.) here

        pass

    

    def encode(self, x):

        """

        Map input trajectories to discrete tokens/codes.

        Input: (Batch, Time, Features)

        Output: (Batch, Time) -> Integer codes

        """

        pass

    

    def decode(self, codes):

        """

        Map discrete tokens back to trajectory space.

        Input: (Batch, Time) -> Integer codes

        Output: (Batch, Time, Features)

        """

        pass

    

    def forward(self, x):

        """

        Standard training forward pass.

        Returns: reconstructed, codes

        """

        pass

    

    def generate(self, n_samples, length):

        """

        Autoregressive generation of new behavior.

        Input: n_samples, length

        Output: (n_samples, length, Features) -> Synthetic trajectories

        """

        # Must implement dynamics (e.g., if VQ-VAE, use a trained prior/Transformer/LSTM here)

        pass

3. Discovery-Specific Loss Function

Python



def discovery_loss(original, reconstructed, codes, model, alpha=1.0, beta=0.25, gamma=0.1, delta=0.5):

    """

    Multi-objective loss for unsupervised discovery.

    This loss function MUST be used regardless of architecture to ensure fair comparison.

    

    Components:

    1. Reconstruction (alpha): Can we decode back?

    2. Commitment (beta): Stable codes during training?

    3. Codebook utilization (gamma): All codes used?

    4. Temporal coherence (delta): Codes persist over time?

    """

    # 1. Reconstruction

    recon_loss = F.mse_loss(reconstructed, original)

    

    # 2. Commitment (Adaptive: if model has it, use it, else 0)

    if hasattr(model, 'commitment_loss'):

        commit_loss = model.commitment_loss

    else:

        commit_loss = torch.tensor(0.0).to(original.device)

    

    # 3. Codebook utilization (maximize entropy of code usage)

    # Flattens batch/time dimensions

    code_probs = torch.bincount(codes.flatten()) / codes.numel()

    code_probs = code_probs + 1e-10  # numerical stability

    codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))

    # We want to MAXIMIZE entropy, so we minimize negative entropy (or difference from max)

    target_entropy = np.log(len(code_probs)) 

    codebook_loss = target_entropy - codebook_entropy

    

    # 4. Temporal coherence (Penalize rapid flickering)

    # Calculate probability of code switching frame-to-frame

    code_changes = (codes[:, 1:] != codes[:, :-1]).float().mean()

    temporal_loss = code_changes

    

    # Combined

    total_loss = (

        alpha * recon_loss +

        beta * commit_loss +

        gamma * codebook_loss +

        delta * temporal_loss

    )

    

    return total_loss, {

        'total': total_loss.item(),

        'reconstruction': recon_loss.item(),

        'commitment': commit_loss.item(),

        'codebook': codebook_loss.item(),

        'temporal': temporal_loss.item()

    }

4. Failure Mode Detection

Python



class FailureModeDetector:

    """Detect and warn about common failure modes"""

    

    def check_all(self, codes, original, reconstructed, n_codes):

        """Run all checks"""

        issues = []

        if self.check_codebook_collapse(codes, n_codes): issues.append("CODEBOOK_COLLAPSE")

        if self.check_temporal_flickering(codes): issues.append("TEMPORAL_FLICKERING")

        if self.check_degenerate_segmentation(codes): issues.append("DEGENERATE_SEGMENTATION")

        if self.check_poor_reconstruction(original, reconstructed): issues.append("POOR_RECONSTRUCTION")

        return issues

    

    def check_codebook_collapse(self, codes, n_codes):

        unique = len(np.unique(codes.cpu().numpy()))

        return unique < n_codes * 0.10 # Strict threshold

    

    def check_temporal_flickering(self, codes):

        change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()

        return change_rate > 0.5

    

    def check_degenerate_segmentation(self, codes):

        freqs = np.bincount(codes.flatten().cpu().numpy()) / codes.numel()

        return freqs.max() > 0.9 # One code dominates 90% of data

    

    def check_poor_reconstruction(self, original, reconstructed):

        # simple check: is variance of recon << variance of original?

        var_orig = torch.var(original)

        var_recon = torch.var(reconstructed)

        return var_recon < (var_orig * 0.1) # Posterior collapse to mean

5. Training Loop

Python



def train_pipeline(model, data, epochs=50, batch_size=32):

    # Implement standard PyTorch training loop calling discovery_loss

    # Return trained model and history dict

    pass

Phase 2: Intrinsic Evaluation (Standardized)

Implement this exact evaluator. It must run on your DiscoveryPipeline instance.



Python



class IntrinsicEvaluator:

    """

    Standardized evaluation for ANY behavior model.

    """

    

    def evaluate_all(self, model, real_data):

        results = {}

        

        # 1. Reconstruction quality

        results['reconstruction_mse'] = self.compute_reconstruction_mse(model, real_data)

        

        # 2. Code statistics

        codes = model.encode(real_data)

        results['codebook_usage'] = len(np.unique(codes.cpu().numpy())) / model.n_codes

        

        # 3. Temporal properties

        results['mean_bout_length'] = self.compute_mean_bout_length(codes)

        

        # 4. Generative quality (CRITICAL)

        # Generate synthetic data of same shape as real_data

        synthetic_data = model.generate(n_samples=real_data.shape[0], length=real_data.shape[1])

        

        # Compute MMD (Maximum Mean Discrepancy) - Kinematic Distribution match

        results['mmd_score'] = self.compute_mmd(real_data, synthetic_data)

        

        # Compute ACF (Autocorrelation) match - Temporal Dynamics match

        results['acf_error'] = self.compute_acf_error(real_data, synthetic_data)

        

        # 5. Combined Discovery Score

        # We want Low MSE, High Usage, Low MMD, Low ACF Error

        results['discovery_score'] = (

            (1.0 / (results['reconstruction_mse'] + 1e-6)) * 0.3 +

            (results['codebook_usage']) * 0.2 +

            (1.0 / (results['mmd_score'] + 1e-6)) * 0.3 +

            (1.0 / (results['acf_error'] + 1e-6)) * 0.2

        )

        

        return results



    # ... Implement helper methods (compute_mmd, compute_acf_error) here ...

Phase 3: Validation (Ground Truth Reveal)

Now, and only now, use the labels to validate if the discovered structure is meaningful.



Python



class ExtrinsicEvaluator:

    def evaluate_with_labels(self, model, data, labels):

        codes = model.encode(data)

        # ARI: How well do codes match human labels?

        ari = adjusted_rand_score(labels.flatten(), codes.flatten().cpu().numpy())

        return {'ari': ari}

Task Execution

Phase 1: Output the complete, executable Python code for the selected pipeline.

Phase 2: Output the IntrinsicEvaluator code.

Instructions: Explain how to run this script to train the model and print the Intrinsic vs Extrinsic scores.

BEGIN.