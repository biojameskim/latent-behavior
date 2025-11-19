# Discovery Agent: From Notes to ICML Paper

**Goal**: Submit to ICML 2025 (~Feb 1st deadline) with a novel contribution on agentic discovery in unsupervised settings.

## 🎯 Core Idea

**Problem**: In true discovery settings (new datasets, no labels), scientists can't use AutoML because there's no metric to optimize. They need:
1. Multiple diverse hypotheses (not a single "best" solution)
2. Proxy objectives (intrinsic metrics) that correlate with meaningfulness
3. Interpretable segmentations

**Solution**: Discovery agents that:
- Optimize intrinsic metrics (reconstruction, code balance, temporal coherence, stability)
- Generate diverse portfolios (5-10 hypotheses with NMI < 0.3)
- Validate on known datasets that intrinsic → extrinsic correlation holds

## ✅ What's Been Built

### 1. Evaluation Infrastructure (`flies/evaluation/discovery_metrics.py`)

**Intrinsic Metrics** (no labels needed):
- Code distribution balance (entropy, codebook utilization)
- Temporal coherence (mean bout length, transition rate)
- Reconstruction quality (MSE, R²)
- Stability (cross-split ARI, seed stability ARI)
- Combined intrinsic score (weighted combination)

**Extrinsic Metrics** (validation with labels):
- Rediscovery (ARI/NMI with ground truth)
- Classification accuracy (logistic regression: codes → labels)
- Forecasting (HMM likelihood, bigram next-token accuracy)

**Portfolio Metrics**:
- Diversity (pairwise NMI between segmentations)
- Coverage (what fraction of true behaviors are captured?)
- Individual quality (mean/max ARI across portfolio)

**Key Functions**:
```python
# Evaluate any segmentation method
results = evaluate_discovery_pipeline(
    codes=discovered_codes,
    labels=ground_truth_labels,  # optional
    reconstruction=reconstructed_data,  # optional
    codes_split2=codes_from_split2,  # for stability
    include_forecasting=True
)

# Returns: {'intrinsic': {...}, 'extrinsic': {...}, 'forecasting': {...}}
```

```python
# Evaluate a portfolio of diverse methods
portfolio_results = evaluate_portfolio(
    portfolio_codes=[method1_codes, method2_codes, method3_codes],
    portfolio_names=['PCA+k-means', 'VQ-VAE', 'MoSeq'],
    labels=ground_truth_labels
)

# Returns: diversity metrics + coverage + individual quality
```

## 📋 Next Steps (10 Weeks to Submission)

### Week 1: Complete Evaluation Setup (THIS WEEK)

**Action Items**:
1. Install dependencies:
   ```bash
   conda activate lat-beh
   pip install scikit-learn hmmlearn
   ```

2. Test the evaluation module:
   ```bash
   python flies/evaluation/discovery_metrics.py
   ```

3. Apply to your existing VQ-VAE models:
   ```python
   # Example: evaluate your trained VQ-VAE
   from flies.evaluation.discovery_metrics import evaluate_discovery_pipeline

   # Load your model's codes
   codes = ...  # from your VQ-VAE

   # Compute all metrics
   results = evaluate_discovery_pipeline(
       codes=codes,
       reconstruction=your_reconstruction,
       original_data=your_data,
   )

   print(f"Intrinsic score: {results['intrinsic']['combined_score']:.3f}")
   print(f"Code entropy: {results['intrinsic']['code_entropy_normalized']:.3f}")
   print(f"Mean bout length: {results['intrinsic']['mean_bout_length']:.1f}")
   ```

**Deliverable**: Evaluation working on your existing models

---

### Week 2: Get MABe22/CalMS21 Data

**Action Items**:
1. Download MABe22 competition data:
   - URL: https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022
   - Focus on mouse triplet dataset (has annotations)

2. Download CalMS21:
   - URL: https://data.caltech.edu/records/1991
   - Task 1 classification data (annotated behaviors)

3. Create ML-ready format:
   ```python
   # data/prepare_mabe.py
   # Convert to: train/val/test splits with keypoints + labels
   # Save as: data/processed/mabe22/{train,val,test}.npz
   ```

4. Verify extrinsic metrics work:
   ```python
   # Test classification accuracy on MABe22
   results = evaluate_discovery_pipeline(
       codes=your_codes,
       labels=mabe_labels,  # ground truth
       train_codes=train_codes,
       train_labels=train_labels
   )

   print(f"Rediscovery ARI: {results['extrinsic']['rediscovery_ari']:.3f}")
   print(f"Classification accuracy: {results['extrinsic']['classification_accuracy']:.3f}")
   ```

**Deliverable**: MABe22 data ready + extrinsic metrics working

---

### Week 3-4: Baseline Methods

Implement 3-5 simple baseline methods to establish the search space:

```python
# flies/methods/baselines.py

class PCAKMeans:
    """PCA feature extraction + k-means clustering"""
    def __init__(self, n_components=10, n_clusters=20):
        ...

    def fit(self, keypoints):
        # PCA on keypoints
        # k-means on PCA features
        return codes

class SimpleVQVAE:
    """Your existing VQ-VAE"""
    # Already implemented

class RandomForestClustering:
    """Unsupervised random forest + k-means"""
    ...

class GMMClustering:
    """Gaussian Mixture Model on raw keypoints"""
    ...
```

**Key Experiment**: Run all baselines on MABe22, create comparison table:

| Method | Intrinsic Score | ARI | Classification Acc | Code Entropy |
|--------|----------------|-----|-------------------|--------------|
| PCA+k-means | 0.65 | 0.42 | 0.68 | 2.8 |
| VQ-VAE | 0.72 | 0.55 | 0.74 | 3.1 |
| GMM | 0.58 | 0.38 | 0.62 | 2.5 |

**Deliverable**: Baseline results showing intrinsic vs extrinsic metrics

---

### Week 5-6: Simple Discovery Agent

Build a minimal agent (single hypothesis):

```python
# flies/agent/simple_agent.py

class SimpleDiscoveryAgent:
    def __init__(self, data, intrinsic_metric_fn):
        self.data = data
        self.metric_fn = intrinsic_metric_fn
        self.history = []

    def propose_pipeline(self):
        """Use LLM to propose next pipeline based on history"""
        prompt = f"""
        You are optimizing unsupervised behavior segmentation.

        Search space:
        - Feature extraction: [raw, pca-5, pca-10, pca-20, hand-crafted]
        - Discretization: [kmeans-10, kmeans-20, gmm-15, vqvae-64]

        Previous attempts:
        {self.format_history()}

        Propose a new pipeline that maximizes:
        - Code entropy (balanced usage)
        - Temporal coherence (long bouts)
        - Reconstruction quality

        Output JSON: {{"feature": "pca-10", "discretization": "kmeans-15"}}
        """

        response = call_llm(prompt)
        return parse_json(response)

    def run(self, n_iterations=20):
        for i in range(n_iterations):
            pipeline = self.propose_pipeline()
            codes, recon = execute_pipeline(pipeline, self.data)
            metrics = self.metric_fn(codes, recon)

            self.history.append({
                'pipeline': pipeline,
                'metrics': metrics,
                'intrinsic_score': metrics['combined_score']
            })

            print(f"Iteration {i}: {pipeline} → score={metrics['combined_score']:.3f}")

        # Return best pipeline
        return max(self.history, key=lambda x: x['intrinsic_score'])
```

**Key Experiment**:
- Run agent on MABe22 with ONLY intrinsic metrics (no labels!)
- After convergence, evaluate extrinsic metrics
- Show: "Agent found pipeline with ARI=0.65 using only intrinsic metrics!"

**Deliverable**: Single-hypothesis agent that discovers good solutions without labels

---

### Week 7-8: Portfolio Agent (CORE CONTRIBUTION)

Extend to multi-hypothesis generation:

```python
# flies/agent/portfolio_agent.py

class PortfolioAgent(SimpleDiscoveryAgent):
    def __init__(self, data, intrinsic_metric_fn, portfolio_size=5):
        super().__init__(data, intrinsic_metric_fn)
        self.portfolio = []
        self.portfolio_size = portfolio_size

    def propose_pipeline(self):
        """Propose pipeline DIVERSE from existing portfolio"""
        prompt = f"""
        Current portfolio has {len(self.portfolio)} methods:
        {self.format_portfolio()}

        Generate a NEW pipeline that:
        1. Scores well on intrinsic metrics
        2. Is DIVERSE from existing portfolio (different feature/discretization combos)

        Avoid redundant combinations.
        """
        return parse_llm_response(prompt)

    def update_portfolio(self, candidate_pipeline, candidate_codes, candidate_metrics):
        """Add if diverse AND high quality"""
        # Compute diversity from portfolio
        diversities = [
            normalized_mutual_info_score(candidate_codes, p['codes'])
            for p in self.portfolio
        ]

        min_diversity = min(diversities) if diversities else 0.0
        quality_threshold = 0.6
        diversity_threshold = 0.3  # NMI < 0.3 = diverse

        # Pareto acceptance
        if candidate_metrics['combined_score'] > quality_threshold and min_diversity < diversity_threshold:
            self.portfolio.append({
                'pipeline': candidate_pipeline,
                'codes': candidate_codes,
                'metrics': candidate_metrics
            })

            # Maintain portfolio size
            if len(self.portfolio) > self.portfolio_size:
                self.portfolio = self.pareto_prune(self.portfolio)
```

**Key Experiments**:

1. **Portfolio improves coverage**:
   ```
   Single agent: ARI=0.65
   Portfolio (5 methods): mean ARI=0.58, max ARI=0.72, coverage=0.85
   ```

2. **Diversity matters** (ablation):
   ```
   Portfolio WITH diversity objective: coverage=0.85
   Top-5 by quality (no diversity): coverage=0.62
   ```

**Deliverable**: Portfolio agent + experiments showing diversity → better coverage

---

### Week 9: Key Results & Figures

**Figure 1: Intrinsic → Extrinsic Correlation**
```python
# Plot all pipelines tried by agent
plt.scatter(intrinsic_scores, extrinsic_ari)
plt.xlabel('Intrinsic Score (no labels)')
plt.ylabel('Extrinsic ARI (ground truth)')
plt.title(f'Correlation: R²={r2:.2f}')
# Show: positive correlation proves proxies work!
```

**Figure 2: Portfolio Diversity & Coverage**
```python
# Confusion matrix: portfolio vs ground truth
# Show different portfolio members capture different behaviors
```

**Figure 3: Comparison to Baselines**
```python
# Bar chart: MoSeq, VAME, VQ-VAE, Single Agent, Portfolio Agent
# Metrics: ARI, Classification Acc, Behavior Coverage
```

**Figure 4: Ablations**
```python
# Portfolio size (1, 3, 5, 10) vs coverage
# With/without diversity objective
```

**Deliverable**: 4 camera-ready figures

---

### Week 10: Paper Writing

**Outline**:
1. **Intro**: Discovery vs optimization; why AutoML fails; portfolio output
2. **Method**: Portfolio agent architecture, intrinsic metrics, diversity objective
3. **Experiments**: MABe22/CalMS21 results
4. **Results**: Figures 1-4, comparisons to baselines
5. **Discussion**: Generalization to other discovery domains

**Deliverable**: Full draft ready for submission

---

## 🚀 Start RIGHT NOW

Your immediate next task (30 minutes):

```bash
# 1. Activate environment
conda activate lat-beh

# 2. Install dependencies
pip install scikit-learn hmmlearn

# 3. Test evaluation module
python flies/evaluation/discovery_metrics.py

# 4. Apply to your existing VQ-VAE
# Create a quick script to evaluate your trained models
```

Then come back and tell me:
1. Does the evaluation module work?
2. What's the intrinsic score of your existing VQ-VAE?
3. Do you have access to MABe22 data yet?

Once you confirm these basics work, we'll move to Week 2 (data prep) or Week 3 (baselines) depending on what you have available.

---

## 🔑 Key Insights for Your PI

When you meet with your PI next, emphasize:

1. **ML Contribution** (not just science tool):
   - Novel problem: discovery without ground truth metrics
   - Portfolio output (not single solution) is fundamental design choice
   - Diversity-driven generation with Pareto acceptance

2. **Validation Strategy**:
   - Use datasets with known structure (MABe22) to validate proxies
   - Intrinsic→extrinsic correlation proves method works
   - Then apply to truly novel datasets

3. **Why not generic AutoML?**:
   - AutoML assumes single metric; we have no metric
   - AutoML converges; we explore continuously
   - AutoML finds optimum; we find diverse portfolio

4. **Timeline**:
   - 10 weeks to ICML submission is tight but doable
   - Weeks 1-4: Infrastructure + baselines (foundation)
   - Weeks 5-8: Agent implementation (core contribution)
   - Weeks 9-10: Results + writing (polish)

Good luck! Start with that 30-minute task and report back.
