## Role
You are an expert Computational Neuroscientist and Auto-ML Engineer specialized in unsupervised behavior discovery. You must act as an **Autonomous Discovery Agent** in "Discovery Mode"—optimizing not just for code correctness, but for scientific insight, interpretability, and generative quality.

## The Challenge
You have raw behavioral trajectory data with **zero ground-truth labels**. You don't know:
- How many behaviors exist
- What timescales matter
- What constitutes "meaningful" structure

You must discover generative models that:
1. Segment continuous trajectories into interpretable discrete behaviors
2. Enable synthesis of new, realistic trajectories

## Dataset

**File**: `mabe22_subset_for_claude.npz`
```python
data = np.load('mabe22_subset_for_claude.npz', allow_pickle=True)

# Structure:
- trajectories: (50, 300, 48)  # 50 sequences, 300 frames, 48 features
- labels: (50, 300)  # Ground truth (DO NOT USE until validation!)
- keypoint_vocabulary: 24 keypoint names
- metadata: dataset info
```

**What this represents**:
- Fruit fly (Drosophila) movements during social interactions
- 24 body keypoints tracked in 2D: wings, eyes, legs, body segments
- 10 seconds per sequence at 30 fps
- Behaviors include: walking, turning, grooming, wing extensions, social interactions

**CRITICAL**: Pretend labels don't exist until final validation phase.

---

## Phase 0: Virtual Architecture Search (Structured Reasoning)

**Before writing any code**, perform systematic comparison:

### Compare at Least 5 Architectural Approaches:

**Approach A: VQ-VAE**
- Pros: Discrete latent space, good for segmentation, proven in behavior
- Cons: Can suffer codebook collapse, requires careful tuning
- Best for: Clean segmentation with generation

**Approach B: VAE + k-means**
- Pros: Flexible continuous latent space, simple clustering
- Cons: Two-stage training, k-means arbitrary
- Best for: Quick baseline

**Approach C: Autoencoder + GMM + HMM**
- Pros: Probabilistic, models transitions explicitly
- Cons: Complex training, many hyperparameters
- Best for: When temporal dynamics are critical

**Approach D: Transformer + VQ**
- Pros: Long-range dependencies, attention mechanism
- Cons: Data hungry, computationally expensive
- Best for: Complex sequential structure

**Approach E: Energy-based (like MCD paper)**
- Pros: Theoretically grounded, no model assumptions
- Cons: Complex implementation, harder to train
- Best for: When avoiding model bias is critical

### Evaluation Criteria Matrix:

Create this table comparing all approaches:

| Approach | Interpretability | Temporal Modeling | Robustness | Generative Quality | Training Speed | Implementation Complexity | Overall Score |
|----------|-----------------|-------------------|------------|-------------------|----------------|--------------------------|---------------|
| VQ-VAE | | | | | | | |
| VAE+kmeans | | | | | | | |
| ...fill in... | | | | | | | |

Score each 1-5, provide justification.

### Selection Decision:

Based on your analysis, select **top 3-5 approaches to implement**.

Justify each selection:
- Why this architecture for this data?
- What specific properties make it suitable?
- What are the key hyperparameters to tune?
- What failure modes should you watch for?

---

## Phase 1: Implementation (Complete, Executable Code)

For each selected architecture, implement a **complete pipeline**.

### Required Components:

#### 1. Preprocessing Module
```python
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
```

#### 2. Model Architecture
```python
class DiscoveryPipeline(nn.Module):
    """
    Complete generative model for behavior discovery
    
    Must include:
    - Encoder: trajectories → latent codes
    - Quantizer/Cluster: continuous → discrete
    - Decoder: codes → trajectories
    - Dynamics: model temporal transitions
    - Generator: sample new trajectories
    """
    
    def __init__(self, ...):
        # YOUR ARCHITECTURE HERE
        # Be specific about:
        # - Number of layers
        # - Hidden dimensions
        # - Activation functions
        # - Normalization
        pass
    
    def encode(self, x):
        """trajectories → codes"""
        pass
    
    def decode(self, codes):
        """codes → trajectories"""
        pass
    
    def forward(self, x):
        """Full forward pass"""
        pass
    
    def generate(self, n_samples, length):
        """Sample new trajectories"""
        # Must implement autoregressive generation
        pass
```

#### 3. Discovery-Specific Loss Function
```python
def discovery_loss(original, reconstructed, codes, model, alpha=1.0, beta=0.25, gamma=0.1, delta=0.5):
    """
    Multi-objective loss for unsupervised discovery
    
    Components:
    1. Reconstruction (alpha): Can we decode back?
    2. Commitment (beta): Stable codes during training?
    3. Codebook utilization (gamma): All codes used?
    4. Temporal coherence (delta): Codes persist over time?
    
    Returns:
        total_loss: scalar
        components: dict of individual losses
    """
    # 1. Reconstruction
    recon_loss = F.mse_loss(reconstructed, original)
    
    # 2. Commitment (if using VQ)
    if hasattr(model, 'commitment_loss'):
        commit_loss = model.commitment_loss
    else:
        commit_loss = torch.tensor(0.0)
    
    # 3. Codebook utilization (entropy)
    code_probs = torch.bincount(codes.flatten()) / codes.numel()
    code_probs = code_probs + 1e-10  # numerical stability
    codebook_entropy = -torch.sum(code_probs * torch.log(code_probs))
    target_entropy = np.log(len(code_probs))  # max entropy
    codebook_loss = target_entropy - codebook_entropy
    
    # 4. Temporal coherence
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
```

#### 4. Failure Mode Detection
```python
class FailureModeDetector:
    """Detect and warn about common failure modes"""
    
    def check_all(self, codes, original, reconstructed, n_codes):
        """Run all checks"""
        issues = []
        
        if self.check_codebook_collapse(codes, n_codes):
            issues.append("CODEBOOK_COLLAPSE")
        
        if self.check_temporal_flickering(codes):
            issues.append("TEMPORAL_FLICKERING")
        
        if self.check_degenerate_segmentation(codes):
            issues.append("DEGENERATE_SEGMENTATION")
        
        if self.check_poor_reconstruction(original, reconstructed):
            issues.append("POOR_RECONSTRUCTION")
        
        return issues
    
    def check_codebook_collapse(self, codes, n_codes):
        unique = len(np.unique(codes))
        return unique < n_codes * 0.5
    
    def check_temporal_flickering(self, codes):
        change_rate = (codes[:, 1:] != codes[:, :-1]).float().mean()
        return change_rate > 0.5
    
    def check_degenerate_segmentation(self, codes):
        freqs = np.bincount(codes.flatten()) / codes.numel()
        return freqs.max() > 0.8
    
    def check_poor_reconstruction(self, original, reconstructed):
        r2 = r2_score(original.flatten().cpu(), reconstructed.flatten().cpu())
        return r2 < 0.5
```

#### 5. Training Loop with Monitoring
```python
def train_pipeline(model, data, epochs=50, batch_size=32):
    """
    Complete training with monitoring and early stopping
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    detector = FailureModeDetector()
    history = {'loss': [], 'issues': []}
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch in DataLoader(data, batch_size=batch_size):
            optimizer.zero_grad()
            
            # Forward
            reconstructed, codes = model(batch)
            
            # Loss
            loss, components = discovery_loss(batch, reconstructed, codes, model)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(components)
        
        # Check for issues
        with torch.no_grad():
            all_codes = model.encode(data)
            all_recon = model.decode(all_codes)
            issues = detector.check_all(all_codes, data, all_recon, model.n_codes)
        
        # Log
        avg_loss = np.mean([l['total'] for l in epoch_losses])
        history['loss'].append(avg_loss)
        history['issues'].append(issues)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Issues={issues}")
        
        # Early stopping if severe issues persist
        if len(issues) > 2 and epoch > 10:
            print("WARNING: Multiple failure modes detected, consider adjusting hyperparameters")
    
    return model, history
```

---

## Phase 2: Intrinsic Evaluation (No Labels!)

For each trained pipeline, compute comprehensive intrinsic metrics:

### Metrics to Implement:
```python
class IntrinsicEvaluator:
    """
    Evaluate generative quality without ground truth
    """
    
    def evaluate_all(self, model, real_data):
        """Compute all intrinsic metrics"""
        results = {}
        
        # 1. Reconstruction quality
        results['reconstruction_r2'] = self.compute_reconstruction_quality(model, real_data)
        
        # 2. Code statistics
        results['code_entropy'] = self.compute_code_entropy(model, real_data)
        results['codebook_utilization'] = self.compute_codebook_utilization(model, real_data)
        
        # 3. Temporal properties
        results['mean_bout_length'] = self.compute_mean_bout_length(model, real_data)
        results['transition_rate'] = self.compute_transition_rate(model, real_data)
        
        # 4. Stability
        results['cross_split_ari'] = self.compute_cross_split_stability(model, real_data)
        
        # 5. Generative quality (CRITICAL!)
        synthetic_data = model.generate(n_samples=50, length=300)
        results['mmd'] = self.compute_mmd(real_data, synthetic_data)
        results['acf_error'] = self.compute_acf_similarity(real_data, synthetic_data)
        results['discriminator_acc'] = self.train_discriminator(real_data, synthetic_data)
        
        # 6. Combined score
        results['intrinsic_score'] = self.compute_combined_score(results)
        
        return results
    
    def compute_mmd(self, real, synthetic, kernel='rbf'):
        """Maximum Mean Discrepancy"""
        # YOUR IMPLEMENTATION
        pass
    
    def compute_acf_similarity(self, real, synthetic, max_lag=50):
        """Autocorrelation function comparison"""
        # YOUR IMPLEMENTATION
        pass
    
    def train_discriminator(self, real, synthetic, epochs=10):
        """Can discriminator tell real from fake?"""
        # Want accuracy ~50% (indistinguishable)
        # YOUR IMPLEMENTATION
        pass
    
    def compute_combined_score(self, metrics):
        """Weighted combination of intrinsic metrics"""
        score = (
            0.20 * metrics['reconstruction_r2'] +
            0.15 * metrics['code_entropy'] / np.log(n_codes) +  # normalize
            0.15 * min(metrics['mean_bout_length'] / 10, 1.0) +  # target ~10 frames
            0.25 * metrics['cross_split_ari'] +
            0.25 * (1 - metrics['mmd']) +  # low MMD is good
            0.10 * (0.5 - abs(0.5 - metrics['discriminator_acc']))  # want 0.5
        )
        return score
```

---

## Phase 3: Portfolio Selection

Select top 5 pipelines based on:
1. High intrinsic scores
2. Diversity (different segmentations)
```python
def select_portfolio(pipelines, evaluations, data, k=5):
    """
    Select diverse, high-quality portfolio
    
    Returns:
        selected: list of k pipeline indices
    """
    # Sort by intrinsic score
    sorted_indices = np.argsort([e['intrinsic_score'] for e in evaluations])[::-1]
    
    # Greedy diversity selection
    selected = [sorted_indices[0]]  # Take best
    
    for idx in sorted_indices[1:]:
        if len(selected) >= k:
            break
        
        # Compute diversity with existing portfolio
        min_diversity = float('inf')
        for sel_idx in selected:
            codes_new = pipelines[idx].encode(data)
            codes_existing = pipelines[sel_idx].encode(data)
            diversity = 1 - adjusted_rand_score(codes_new, codes_existing)
            min_diversity = min(min_diversity, diversity)
        
        # Accept if diverse enough (ARI < 0.7 with all existing)
        if min_diversity > 0.3:
            selected.append(idx)
    
    return selected
```

---

## Phase 4: Validation (NOW Use Labels)

For pipelines in portfolio, compute extrinsic metrics:
```python
class ExtrinsicEvaluator:
    """Evaluate using ground truth labels"""
    
    def evaluate_all(self, model, data, labels):
        results = {}
        
        # 1. Alignment with ground truth
        codes = model.encode(data)
        results['ari'] = adjusted_rand_score(labels.flatten(), codes.flatten())
        results['nmi'] = normalized_mutual_info_score(labels.flatten(), codes.flatten())
        
        # 2. Downstream classification
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)
        
        # Flatten for classification
        X = codes.reshape(-1, 1)
        y = labels.flatten()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf.fit(X_train, y_train)
        results['classification_acc'] = clf.score(X_test, y_test)
        
        # 3. Behavior statistics match
        real_bouts = self.compute_bout_lengths(labels)
        discovered_bouts = self.compute_bout_lengths(codes)
        results['bout_length_ks'] = ks_2samp(real_bouts, discovered_bouts).pvalue
        
        return results
```

---

## Phase 5: Analysis and Reporting

### Required Deliverables:

1. **Pipeline Comparison Table** (CSV):
   - All pipelines tried
   - Intrinsic scores
   - Extrinsic scores (for portfolio)
   - Training time
   - Failure modes encountered

2. **Intrinsic-Extrinsic Correlation Plot**:
   - X-axis: Intrinsic score
   - Y-axis: Extrinsic score (ARI or classification accuracy)
   - Each point = one pipeline
   - Show Pearson correlation coefficient
   - **KEY VALIDATION**: High correlation means intrinsic metrics work!

3. **Portfolio Visualization**:
   - t-SNE of discovered codes for each portfolio pipeline
   - Show diversity visually
   - Highlight where they agree/disagree

4. **Generation Quality**:
   - Plot example real vs. synthetic trajectories
   - Show ACF comparison
   - Show MMD scores
   - Visual inspection: do they look realistic?

5. **Final Report** (`discovery_report.md`):
   - Virtual architecture search reasoning
   - Which pipelines worked best and why
   - Failure modes encountered and how handled
   - Key findings about the data
   - Recommendations for neuroscientists

---

## Constraints and Requirements

### Must-Haves:
✅ Complete, executable code (no placeholders)
✅ At least 3 different architectural approaches implemented
✅ All intrinsic metrics implemented (including MMD, ACF)
✅ Failure mode detection and handling
✅ Portfolio selection with diversity
✅ Validation showing intrinsic-extrinsic correlation

### Forbidden:
❌ Using labels during discovery phase
❌ Generic code ("insert your model here")
❌ Single architecture without comparison
❌ Skipping generation quality evaluation
❌ No justification for hyperparameter choices

### Hyperparameters:
Choose based on your virtual architecture search, not random defaults:
- Window size: justify based on temporal structure
- Codebook size: justify based on expected behavior count
- Latent dimensions: justify based on data complexity
- Learning rate, batch size: standard ranges okay

---

## Meta-Questions (For My Research)

After completing this task, reflect:

1. **Exploration**: Did you systematically explore architectures or gravitate toward familiar ones?
2. **Discovery**: Did intrinsic metrics successfully predict extrinsic quality?
3. **Automation**: What required human judgment vs. what could be automated?
4. **Scaling**: How would this approach scale to larger/more complex datasets?
5. **Limitations**: What did you struggle with? What would a specialized framework add?

---

## Success Criteria

A successful discovery agent:
- ✅ Tries 3-5 diverse architectures (not just one)
- ✅ Uses only intrinsic metrics during search (no label peeking)
- ✅ Maintains portfolio diversity (not all similar)
- ✅ Shows intrinsic-extrinsic correlation >0.6
- ✅ Generates realistic trajectories (discriminator accuracy ~50%, low MMD)
- ✅ Detects and handles failure modes
- ✅ Provides scientific insights (not just code)

---

**BEGIN!**

Start with Phase 0 (virtual architecture search), then systematically work through each phase.