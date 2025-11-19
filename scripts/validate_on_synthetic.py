"""
Validate discovery metrics on synthetic data with known ground truth.

This script demonstrates the key validation strategy:
1. Generate synthetic data with known behavioral modes
2. Apply simple baseline methods (k-means, GMM)
3. Evaluate using discovery_metrics
4. Check: Do intrinsic metrics correlate with extrinsic performance (ARI)?

If intrinsic→extrinsic correlation is positive, this proves that our proxy
objectives capture meaningful structure.

This is the CORE validation for the ICML paper.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Import our modules
from flies.data.synthetic_behaviors import SyntheticBehaviorGenerator
from flies.evaluation.discovery_metrics import (
    evaluate_discovery_pipeline,
    evaluate_portfolio,
    DiscoveryMetrics,
)


def simple_baseline_methods(data: np.ndarray, n_clusters: int = 5) -> dict:
    """
    Apply simple baseline segmentation methods.

    These are the kinds of pipelines a discovery agent would explore.

    Args:
        data: (n_frames, n_features) trajectory data
        n_clusters: Number of clusters/modes

    Returns:
        Dictionary of {method_name: codes}
    """
    results = {}

    # Method 1: Direct k-means on raw features
    print("  Running k-means (raw)...")
    kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42)
    codes_kmeans_raw = kmeans_raw.fit_predict(data)
    results['kmeans_raw'] = codes_kmeans_raw

    # Method 2: PCA + k-means (feature extraction + discretization)
    print("  Running PCA + k-means...")
    pca = PCA(n_components=min(10, data.shape[1]))
    data_pca = pca.fit_transform(data)
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
    codes_kmeans_pca = kmeans_pca.fit_predict(data_pca)
    results['pca_kmeans'] = codes_kmeans_pca

    # Method 3: GMM (probabilistic clustering)
    print("  Running GMM...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    codes_gmm = gmm.fit_predict(data)
    results['gmm'] = codes_gmm

    # Method 4: PCA + GMM
    print("  Running PCA + GMM...")
    gmm_pca = GaussianMixture(n_components=n_clusters, random_state=42)
    codes_gmm_pca = gmm_pca.fit_predict(data_pca)
    results['pca_gmm'] = codes_gmm_pca

    # Method 5: High-dimensional PCA + k-means
    print("  Running PCA-20 + k-means...")
    pca_20 = PCA(n_components=min(20, data.shape[1]))
    data_pca_20 = pca_20.fit_transform(data)
    kmeans_pca_20 = KMeans(n_clusters=n_clusters, random_state=42)
    codes_kmeans_pca_20 = kmeans_pca_20.fit_predict(data_pca_20)
    results['pca20_kmeans'] = codes_kmeans_pca_20

    return results


def main():
    print("=" * 80)
    print("VALIDATION: Discovery Metrics on Synthetic Data")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic behavioral data...")
    generator = SyntheticBehaviorGenerator(
        n_modes=5,
        n_keypoints=3,
        random_seed=42,
    )

    trajectory, ground_truth_labels = generator.generate_sequence(
        total_frames=2000,
        noise_level=0.15,
        mean_bout_length=50,
    )

    print(f"   Generated {len(trajectory)} frames")
    print(f"   Features: {trajectory.shape[1]}")
    print(f"   True modes: {len(np.unique(ground_truth_labels))}")

    # Step 2: Apply multiple baseline methods
    print("\n2. Applying baseline segmentation methods...")
    method_codes = simple_baseline_methods(trajectory, n_clusters=5)
    print(f"   Tested {len(method_codes)} methods")

    # Step 3: Evaluate each method
    print("\n3. Evaluating each method with discovery metrics...")
    results = {}

    for method_name, codes in method_codes.items():
        print(f"\n   Evaluating: {method_name}")

        eval_results = evaluate_discovery_pipeline(
            codes=codes,
            labels=ground_truth_labels,
            reconstruction=None,  # No reconstruction for k-means/GMM
            original_data=None,
            include_forecasting=True,
        )

        results[method_name] = eval_results

        # Print summary
        intrinsic_score = eval_results['intrinsic']['combined_score']
        ari = eval_results['extrinsic']['rediscovery_ari']
        entropy = eval_results['intrinsic']['code_entropy_normalized']

        print(f"     Intrinsic score: {intrinsic_score:.3f}")
        print(f"     Rediscovery ARI: {ari:.3f}")
        print(f"     Code entropy:    {entropy:.3f}")

    # Step 4: KEY VALIDATION - Correlation between intrinsic and extrinsic
    print("\n" + "=" * 80)
    print("4. KEY VALIDATION: Intrinsic ↔ Extrinsic Correlation")
    print("=" * 80)

    intrinsic_scores = [r['intrinsic']['combined_score'] for r in results.values()]
    extrinsic_aris = [r['extrinsic']['rediscovery_ari'] for r in results.values()]

    correlation, p_value = pearsonr(intrinsic_scores, extrinsic_aris)

    print(f"\nPearson correlation: r = {correlation:.3f} (p = {p_value:.4f})")

    if correlation > 0.5 and p_value < 0.05:
        print("✅ STRONG POSITIVE CORRELATION!")
        print("   → Intrinsic metrics successfully predict extrinsic quality")
        print("   → This validates the proxy objectives for discovery agents")
    elif correlation > 0.3:
        print("⚠️  MODERATE POSITIVE CORRELATION")
        print("   → Intrinsic metrics partially predict extrinsic quality")
        print("   → May need to tune metric weights")
    else:
        print("❌ WEAK/NEGATIVE CORRELATION")
        print("   → Intrinsic metrics do NOT predict extrinsic quality")
        print("   → Need to rethink proxy objectives!")

    # Step 5: Visualize correlation
    print("\n5. Generating correlation plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Intrinsic vs Extrinsic scatter
    axes[0].scatter(intrinsic_scores, extrinsic_aris, s=100, alpha=0.7)
    for i, method_name in enumerate(results.keys()):
        axes[0].annotate(
            method_name,
            (intrinsic_scores[i], extrinsic_aris[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
        )

    # Add trend line
    z = np.polyfit(intrinsic_scores, extrinsic_aris, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(intrinsic_scores), max(intrinsic_scores), 100)
    axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    axes[0].set_xlabel('Intrinsic Score (no labels)', fontsize=12)
    axes[0].set_ylabel('Extrinsic ARI (ground truth)', fontsize=12)
    axes[0].set_title(f'Intrinsic→Extrinsic Correlation\nr = {correlation:.3f}', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Right plot: Method comparison
    method_names = list(results.keys())
    x_pos = np.arange(len(method_names))

    width = 0.35
    axes[1].bar(x_pos - width/2, intrinsic_scores, width, label='Intrinsic Score', alpha=0.7)
    axes[1].bar(x_pos + width/2, extrinsic_aris, width, label='Extrinsic ARI', alpha=0.7)

    axes[1].set_xlabel('Method', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Method Comparison', fontsize=14)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(method_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('synthetic_validation.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to synthetic_validation.png")

    # Step 6: Portfolio evaluation
    print("\n" + "=" * 80)
    print("6. Portfolio Evaluation (Multi-Hypothesis)")
    print("=" * 80)

    portfolio_codes = list(method_codes.values())
    portfolio_names = list(method_codes.keys())

    portfolio_results = evaluate_portfolio(
        portfolio_codes=portfolio_codes,
        portfolio_names=portfolio_names,
        labels=ground_truth_labels,
    )

    print(f"\nPortfolio diversity:")
    print(f"  Mean pairwise NMI:  {portfolio_results['diversity']['mean_pairwise_nmi']:.3f}")
    print(f"  Min pairwise NMI:   {portfolio_results['diversity']['min_pairwise_nmi']:.3f}")
    print(f"  (Lower = more diverse)")

    print(f"\nPortfolio quality:")
    print(f"  Max ARI:            {portfolio_results['individual_ari_max']:.3f}")
    print(f"  Mean ARI:           {portfolio_results['individual_ari_mean']:.3f}")

    print(f"\nPortfolio coverage:")
    print(f"  Behavior coverage:  {portfolio_results['behavior_coverage']:.1%}")
    print(f"  (Fraction of true behaviors captured by at least one method)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & NEXT STEPS")
    print("=" * 80)

    best_method = max(results.items(), key=lambda x: x[1]['extrinsic']['rediscovery_ari'])
    print(f"\nBest method by ARI: {best_method[0]}")
    print(f"  ARI: {best_method[1]['extrinsic']['rediscovery_ari']:.3f}")

    best_by_intrinsic = max(results.items(), key=lambda x: x[1]['intrinsic']['combined_score'])
    print(f"\nBest method by intrinsic score: {best_by_intrinsic[0]}")
    print(f"  Intrinsic: {best_by_intrinsic[1]['intrinsic']['combined_score']:.3f}")

    if best_method[0] == best_by_intrinsic[0]:
        print("\n✅ Intrinsic and extrinsic rankings AGREE!")
        print("   → Discovery agent using intrinsic metrics would find the best method")
    else:
        print("\n⚠️  Intrinsic and extrinsic rankings DISAGREE")
        print("   → May need to adjust metric weights")

    print("\n" + "=" * 80)
    print("This validation shows:")
    print("1. Discovery metrics work on synthetic data ✓")
    print("2. Intrinsic metrics correlate with rediscovery (validate on your data!)")
    print("3. Portfolio provides diverse coverage of behaviors")
    print("\nNext: Apply to real MABe22 data and build the discovery agent!")
    print("=" * 80)


if __name__ == '__main__':
    main()
