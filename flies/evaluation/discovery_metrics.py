"""
Discovery-specific evaluation metrics for unsupervised behavior segmentation.

This module extends compare_models.py with metrics needed for discovery agents:
- Intrinsic metrics: stability, diversity, temporal coherence
- Extrinsic metrics: classification accuracy, ARI, forecasting

These metrics allow validating that intrinsic objectives correlate with
extrinsic performance on datasets with known ground truth.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class DiscoveryMetrics:
    """
    Comprehensive metrics for evaluating unsupervised behavior discovery.

    Philosophy:
    - Intrinsic metrics: Can be computed without labels (used during discovery)
    - Extrinsic metrics: Require labels (used for validation on known datasets)
    """

    @staticmethod
    def compute_intrinsic_metrics(
        codes: np.ndarray,
        reconstruction: Optional[np.ndarray] = None,
        original_data: Optional[np.ndarray] = None,
        codes_split2: Optional[np.ndarray] = None,
        codes_seed2: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute intrinsic quality metrics (no labels required).

        Args:
            codes: (n_frames,) discrete behavior codes
            reconstruction: (n_frames, n_features) reconstructed data
            original_data: (n_frames, n_features) original data
            codes_split2: codes from second data split (for stability)
            codes_seed2: codes from different random seed (for stability)

        Returns:
            Dictionary of intrinsic metrics
        """
        metrics = {}

        # 1. Code distribution balance
        code_counts = np.bincount(codes)
        total_codes = len(code_counts)
        used_codes = np.sum(code_counts > 0)

        # Entropy (higher = more balanced usage)
        code_probs = code_counts / code_counts.sum()
        code_entropy = entropy(code_probs + 1e-10)
        max_entropy = np.log(total_codes)

        metrics['code_entropy'] = code_entropy
        metrics['code_entropy_normalized'] = code_entropy / max_entropy if max_entropy > 0 else 0.0
        metrics['codebook_utilization'] = used_codes / total_codes
        metrics['num_codes_used'] = used_codes
        metrics['num_codes_total'] = total_codes

        # 2. Temporal coherence
        # Mean bout length (how long behaviors persist)
        bout_lengths = []
        current_bout = 1
        for i in range(1, len(codes)):
            if codes[i] == codes[i-1]:
                current_bout += 1
            else:
                bout_lengths.append(current_bout)
                current_bout = 1
        bout_lengths.append(current_bout)

        metrics['mean_bout_length'] = np.mean(bout_lengths)
        metrics['median_bout_length'] = np.median(bout_lengths)
        metrics['std_bout_length'] = np.std(bout_lengths)

        # Transition rate (lower = more stable)
        transitions = np.sum(codes[1:] != codes[:-1])
        metrics['transition_rate'] = transitions / (len(codes) - 1)

        # 3. Reconstruction quality (if provided)
        if reconstruction is not None and original_data is not None:
            mse = np.mean((reconstruction - original_data) ** 2)
            metrics['reconstruction_mse'] = mse

            # R^2 score
            ss_res = np.sum((original_data - reconstruction) ** 2)
            ss_tot = np.sum((original_data - np.mean(original_data, axis=0)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics['reconstruction_r2'] = r2

        # 4. Stability metrics (if multiple splits/seeds provided)
        if codes_split2 is not None:
            # Cross-split ARI (same method, different data)
            min_len = min(len(codes), len(codes_split2))
            ari = adjusted_rand_score(codes[:min_len], codes_split2[:min_len])
            metrics['cross_split_ari'] = ari

        if codes_seed2 is not None:
            # Seed stability ARI (same data, different initialization)
            min_len = min(len(codes), len(codes_seed2))
            ari = adjusted_rand_score(codes[:min_len], codes_seed2[:min_len])
            metrics['seed_stability_ari'] = ari

        return metrics

    @staticmethod
    def compute_extrinsic_metrics(
        codes: np.ndarray,
        labels: np.ndarray,
        train_codes: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute extrinsic metrics (requires ground truth labels).

        Used for validation on datasets like MABe22/CalMS21 where we know
        true behaviors. Shows whether intrinsic metrics correlate with
        meaningful structure.

        Args:
            codes: (n_frames,) discovered behavior codes
            labels: (n_frames,) ground truth behavior labels
            train_codes: training codes for supervised evaluation
            train_labels: training labels for supervised evaluation

        Returns:
            Dictionary of extrinsic metrics
        """
        metrics = {}

        # 1. Rediscovery: ARI with ground truth
        # Measures how well discovered codes align with known behaviors
        ari = adjusted_rand_score(labels, codes)
        nmi = normalized_mutual_info_score(labels, codes)

        metrics['rediscovery_ari'] = ari
        metrics['rediscovery_nmi'] = nmi

        # 2. Downstream classification
        # Train simple classifier: codes → labels
        if train_codes is not None and train_labels is not None:
            clf = LogisticRegression(max_iter=1000, random_state=42)

            # Reshape for sklearn
            X_train = train_codes.reshape(-1, 1)
            X_test = codes.reshape(-1, 1)

            clf.fit(X_train, train_labels)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(labels, y_pred)
            f1_macro = f1_score(labels, y_pred, average='macro')
            f1_weighted = f1_score(labels, y_pred, average='weighted')

            metrics['classification_accuracy'] = accuracy
            metrics['classification_f1_macro'] = f1_macro
            metrics['classification_f1_weighted'] = f1_weighted

        return metrics

    @staticmethod
    def compute_forecasting_metrics(
        codes: np.ndarray,
        train_codes: Optional[np.ndarray] = None,
        method: str = 'hmm',
    ) -> Dict[str, float]:
        """
        Evaluate forecasting performance using discovered codes.

        Good behavioral segmentations should have predictable dynamics.

        Args:
            codes: (n_frames,) test codes for evaluation
            train_codes: training codes to fit forecasting model
            method: 'hmm' or 'bigram'

        Returns:
            Dictionary of forecasting metrics
        """
        metrics = {}

        if method == 'hmm':
            try:
                from hmmlearn import hmm

                # Fit HMM on training codes
                if train_codes is None:
                    train_codes = codes[:int(0.8 * len(codes))]
                    test_codes = codes[int(0.8 * len(codes)):]
                else:
                    test_codes = codes

                # Reshape for hmmlearn
                X_train = train_codes.reshape(-1, 1)
                X_test = test_codes.reshape(-1, 1)

                # Fit HMM
                n_states = len(np.unique(train_codes))
                model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
                model.fit(X_train)

                # Compute log-likelihood on test set
                log_likelihood = model.score(X_test)
                metrics['hmm_log_likelihood'] = log_likelihood
                metrics['hmm_perplexity'] = np.exp(-log_likelihood / len(test_codes))

            except ImportError:
                print("Warning: hmmlearn not installed, skipping HMM forecasting")

        elif method == 'bigram':
            # Simple bigram model (next-token prediction)
            if train_codes is None:
                train_codes = codes[:int(0.8 * len(codes))]
                test_codes = codes[int(0.8 * len(codes)):]
            else:
                test_codes = codes

            # Build bigram transition matrix
            n_codes = max(np.max(train_codes), np.max(test_codes)) + 1
            transition_matrix = np.zeros((n_codes, n_codes))

            for i in range(len(train_codes) - 1):
                transition_matrix[train_codes[i], train_codes[i+1]] += 1

            # Normalize with smoothing
            transition_matrix += 0.01  # Laplace smoothing
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

            # Compute next-token accuracy on test set
            correct = 0
            total = 0
            for i in range(len(test_codes) - 1):
                current = test_codes[i]
                next_actual = test_codes[i+1]
                next_pred = np.argmax(transition_matrix[current])
                if next_pred == next_actual:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0.0
            metrics['bigram_next_token_accuracy'] = accuracy

            # Compute log-likelihood
            log_likelihood = 0.0
            for i in range(len(test_codes) - 1):
                current = test_codes[i]
                next_code = test_codes[i+1]
                prob = transition_matrix[current, next_code]
                log_likelihood += np.log(prob + 1e-10)

            metrics['bigram_log_likelihood'] = log_likelihood / (len(test_codes) - 1)

        return metrics

    @staticmethod
    def compute_diversity_metrics(
        code_sets: List[np.ndarray],
        names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Measure diversity across multiple segmentations (portfolio evaluation).

        For discovery agents that output multiple hypotheses, we want them
        to be diverse (low NMI) while each being individually high quality.

        Args:
            code_sets: List of (n_frames,) code arrays from different methods
            names: Optional names for each code set

        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}

        if len(code_sets) < 2:
            return {'error': 'Need at least 2 code sets for diversity metrics'}

        # Pairwise NMI matrix
        n = len(code_sets)
        nmi_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                min_len = min(len(code_sets[i]), len(code_sets[j]))
                nmi = normalized_mutual_info_score(
                    code_sets[i][:min_len],
                    code_sets[j][:min_len],
                )
                nmi_matrix[i, j] = nmi
                nmi_matrix[j, i] = nmi

        # Summary statistics
        # Lower NMI = more diverse portfolio
        upper_triangle = nmi_matrix[np.triu_indices(n, k=1)]
        metrics['mean_pairwise_nmi'] = np.mean(upper_triangle)
        metrics['median_pairwise_nmi'] = np.median(upper_triangle)
        metrics['min_pairwise_nmi'] = np.min(upper_triangle)
        metrics['max_pairwise_nmi'] = np.max(upper_triangle)

        # Store full matrix if names provided
        if names is not None:
            metrics['nmi_matrix'] = {
                'names': names,
                'matrix': nmi_matrix.tolist(),
            }

        return metrics

    @staticmethod
    def compute_combined_score(
        intrinsic_metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Combine multiple intrinsic metrics into single score.

        This is the objective function for discovery agents.

        Default weighting:
        - Code balance (entropy): 0.3
        - Temporal coherence (bout length): 0.2
        - Reconstruction quality: 0.3
        - Stability: 0.2

        Args:
            intrinsic_metrics: Output from compute_intrinsic_metrics()
            weights: Custom weights for each metric

        Returns:
            Combined score (higher = better)
        """
        if weights is None:
            weights = {
                'code_entropy_normalized': 0.3,
                'mean_bout_length': 0.2,
                'reconstruction_r2': 0.3,
                'seed_stability_ari': 0.2,
            }

        score = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():
            if metric_name in intrinsic_metrics:
                value = intrinsic_metrics[metric_name]

                # Normalize to [0, 1] range based on metric type
                if metric_name == 'mean_bout_length':
                    # Normalize bout length (assume reasonable range is 1-50)
                    normalized_value = min(value / 50.0, 1.0)
                elif metric_name in ['reconstruction_r2', 'seed_stability_ari', 'cross_split_ari']:
                    # Already in [0, 1]
                    normalized_value = max(0, value)  # Clip negative R^2
                elif metric_name == 'code_entropy_normalized':
                    # Already normalized
                    normalized_value = value
                else:
                    normalized_value = value

                score += weight * normalized_value
                total_weight += weight

        # Normalize by total weight used
        if total_weight > 0:
            score = score / total_weight

        return score


def evaluate_discovery_pipeline(
    codes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    reconstruction: Optional[np.ndarray] = None,
    original_data: Optional[np.ndarray] = None,
    codes_split2: Optional[np.ndarray] = None,
    codes_seed2: Optional[np.ndarray] = None,
    train_codes: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    include_forecasting: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of a single discovery pipeline.

    This is the main function to use for evaluating any behavior segmentation method.

    Args:
        codes: Discovered behavior codes
        labels: Ground truth labels (optional, for extrinsic eval)
        reconstruction: Reconstructed data (optional)
        original_data: Original data (optional)
        codes_split2: Codes from second split (for stability)
        codes_seed2: Codes from second seed (for stability)
        train_codes: Training codes (for supervised eval)
        train_labels: Training labels (for supervised eval)
        include_forecasting: Whether to compute forecasting metrics

    Returns:
        Dictionary with 'intrinsic' and 'extrinsic' metric dicts
    """
    results = {}

    # Intrinsic metrics (always computed)
    results['intrinsic'] = DiscoveryMetrics.compute_intrinsic_metrics(
        codes=codes,
        reconstruction=reconstruction,
        original_data=original_data,
        codes_split2=codes_split2,
        codes_seed2=codes_seed2,
    )

    # Combined intrinsic score
    results['intrinsic']['combined_score'] = DiscoveryMetrics.compute_combined_score(
        results['intrinsic']
    )

    # Extrinsic metrics (if labels provided)
    if labels is not None:
        results['extrinsic'] = DiscoveryMetrics.compute_extrinsic_metrics(
            codes=codes,
            labels=labels,
            train_codes=train_codes,
            train_labels=train_labels,
        )

    # Forecasting metrics (if requested)
    if include_forecasting:
        results['forecasting'] = DiscoveryMetrics.compute_forecasting_metrics(
            codes=codes,
            train_codes=train_codes,
            method='bigram',
        )

    return results


def evaluate_portfolio(
    portfolio_codes: List[np.ndarray],
    portfolio_names: Optional[List[str]] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate a portfolio of diverse segmentations.

    Key metrics:
    - Diversity: How different are the segmentations?
    - Quality: How well does each perform individually?
    - Coverage: Do they cover different aspects of ground truth?

    Args:
        portfolio_codes: List of discovered code arrays
        portfolio_names: Names for each portfolio member
        labels: Ground truth labels (for coverage analysis)

    Returns:
        Portfolio-level metrics
    """
    results = {}

    # Diversity metrics
    diversity = DiscoveryMetrics.compute_diversity_metrics(
        portfolio_codes,
        names=portfolio_names,
    )
    results['diversity'] = diversity

    # Individual quality metrics
    if labels is not None:
        individual_aris = []
        for codes in portfolio_codes:
            min_len = min(len(codes), len(labels))
            ari = adjusted_rand_score(labels[:min_len], codes[:min_len])
            individual_aris.append(ari)

        results['individual_ari_mean'] = np.mean(individual_aris)
        results['individual_ari_max'] = np.max(individual_aris)
        results['individual_ari_std'] = np.std(individual_aris)

        # Coverage: what fraction of ground truth behaviors are captured?
        # by at least one portfolio member
        unique_labels = np.unique(labels)
        covered_behaviors = set()

        for codes in portfolio_codes:
            min_len = min(len(codes), len(labels))
            # For each ground truth behavior, check if it's captured by this segmentation
            for true_behavior in unique_labels:
                behavior_mask = labels[:min_len] == true_behavior
                if behavior_mask.sum() > 0:
                    # Find which discovered codes overlap with this true behavior
                    overlapping_codes = codes[:min_len][behavior_mask]
                    most_common_code = np.bincount(overlapping_codes).argmax()
                    # If >50% of this true behavior maps to one discovered code, consider it covered
                    overlap_pct = (overlapping_codes == most_common_code).sum() / len(overlapping_codes)
                    if overlap_pct > 0.5:
                        covered_behaviors.add(true_behavior)

        results['behavior_coverage'] = len(covered_behaviors) / len(unique_labels)

    return results


if __name__ == '__main__':
    # Example usage
    print("Discovery Metrics Module")
    print("=" * 80)

    # Generate synthetic example
    np.random.seed(42)
    n_frames = 1000

    # Simulate discovered codes
    codes = np.random.randint(0, 10, size=n_frames)

    # Simulate ground truth labels (for extrinsic eval)
    labels = np.random.randint(0, 5, size=n_frames)

    # Evaluate
    results = evaluate_discovery_pipeline(
        codes=codes,
        labels=labels,
        include_forecasting=True,
    )

    print("\nIntrinsic Metrics:")
    for k, v in results['intrinsic'].items():
        print(f"  {k}: {v:.4f}")

    if 'extrinsic' in results:
        print("\nExtrinsic Metrics:")
        for k, v in results['extrinsic'].items():
            print(f"  {k}: {v:.4f}")

    if 'forecasting' in results:
        print("\nForecasting Metrics:")
        for k, v in results['forecasting'].items():
            print(f"  {k}: {v:.4f}")

    # Example portfolio evaluation
    print("\n" + "=" * 80)
    print("Portfolio Evaluation Example")
    print("=" * 80)

    portfolio = [
        np.random.randint(0, 10, size=n_frames),
        np.random.randint(0, 8, size=n_frames),
        np.random.randint(0, 12, size=n_frames),
    ]

    portfolio_results = evaluate_portfolio(
        portfolio,
        portfolio_names=['Method A', 'Method B', 'Method C'],
        labels=labels,
    )

    print("\nPortfolio Metrics:")
    for k, v in portfolio_results.items():
        if k != 'diversity':
            print(f"  {k}: {v:.4f}")

    print("\nDiversity Metrics:")
    for k, v in portfolio_results['diversity'].items():
        if k != 'nmi_matrix':
            print(f"  {k}: {v:.4f}")
