import numpy as np
import torch
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import BehaviorPreprocessor, DiscoveryPipeline, train_pipeline, FailureModeDetector
from evaluator import IntrinsicEvaluator, ExtrinsicEvaluator

def main():
    print("Loading data...")
    # Load data
    data_path = '/share/j_sun/jjk297/data/fly_data/mabe22_subset_for_claude.npz'
    data = np.load(data_path, allow_pickle=True)
    trajectories = data['trajectories'] # (50, 300, 48)
    labels = data['labels'] # (50, 300)
    
    print(f"Data shape: {trajectories.shape}")
    
    # 1. Preprocess
    print("Preprocessing...")
    preprocessor = BehaviorPreprocessor()
    processed_data = preprocessor.preprocess(trajectories)
    
    # Convert to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_data = processed_data.to(device)
    
    # 2. Initialize Model
    print("Initializing GNN-VQ-VAE...")
    # Input dim is 2 * original features (pos + vel) = 96
    model = DiscoveryPipeline(input_dim=96, num_nodes=24, hidden_dim=64, latent_dim=32, num_embeddings=64).to(device)
    
    # 3. Train
    print("Training...")
    model, history = train_pipeline(model, processed_data, epochs=100, batch_size=16)
    
    # 4. Intrinsic Evaluation
    print("\nRunning Intrinsic Evaluation...")
    evaluator = IntrinsicEvaluator()
    intrinsic_results = evaluator.evaluate_all(model, processed_data)
    
    print("\nIntrinsic Metrics:")
    for k, v in intrinsic_results.items():
        print(f"{k}: {v:.4f}")
        
    # 5. Failure Mode Detection
    print("\nChecking for Failure Modes...")
    detector = FailureModeDetector()
    with torch.no_grad():
        recon, codes = model(processed_data)
    issues = detector.check_all(codes, processed_data, recon, model.n_codes)
    if issues:
        print(f"WARNING: Detected failure modes: {issues}")
    else:
        print("No obvious failure modes detected.")
        
    # 6. Extrinsic Evaluation (Validation)
    print("\nRunning Extrinsic Evaluation (Ground Truth Validation)...")
    ext_evaluator = ExtrinsicEvaluator()
    extrinsic_results = ext_evaluator.evaluate_with_labels(model, processed_data, labels)
    
    print("\nExtrinsic Metrics:")
    print(f"Adjusted Rand Index (ARI): {extrinsic_results['ari']:.4f}")

if __name__ == "__main__":
    main()
