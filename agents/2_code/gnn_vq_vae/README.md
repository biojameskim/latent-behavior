# GNN-VQ-VAE Behavior Discovery Pipeline

This directory contains the implementation of the "Quantized Body-Graph" (GNN-VQ-VAE) architecture for unsupervised behavior discovery in Drosophila.

## Directory Structure

*   `pipeline.py`: Contains the core `DiscoveryPipeline` model (GNN encoder, VQ bottleneck, Transformer dynamics), `BehaviorPreprocessor`, `discovery_loss`, and `FailureModeDetector`.
*   `evaluator.py`: Contains `IntrinsicEvaluator` (unsupervised metrics) and `ExtrinsicEvaluator` (supervised validation).
*   `main.py`: The execution script that ties everything together.

## How to Run

1.  **Environment Setup**: Ensure you have the required dependencies installed (PyTorch, NumPy, Scikit-learn).
    ```bash
    pip install torch numpy scikit-learn
    ```

2.  **Execute the Pipeline**: Run the `main.py` script. This script will:
    *   Load the dataset (`mabe22_subset_for_claude.npz`).
    *   Preprocess the trajectories (ego-centric alignment, velocity computation).
    *   Train the GNN-VQ-VAE model for 50 epochs.
    *   Compute **Intrinsic Metrics** (Reconstruction MSE, Codebook Usage, MMD, ACF Error).
    *   Run **Failure Mode Detection** to check for codebook collapse or flickering.
    *   Compute **Extrinsic Metrics** (Adjusted Rand Index against ground truth labels) for validation.

    ```bash
    python3 main.py
    ```

## Output Interpretation

The script will print the following metrics to the console:

*   **Intrinsic Metrics**:
    *   `reconstruction_mse`: Lower is better. Measures how well the model preserves motion details.
    *   `codebook_usage`: Higher is better (max 1.0). Indicates diversity of discovered behaviors.
    *   `mmd_score`: Lower is better. Measures generative quality (kinematic distribution match).
    *   `acf_error`: Lower is better. Measures temporal dynamics match.
    *   `discovery_score`: Combined score used for model selection.

*   **Extrinsic Metrics**:
    *   `ari` (Adjusted Rand Index): 0.0 = Random, 1.0 = Perfect match with human labels. This reveals how biologically meaningful the discovered "words" are.
