"""
Quick test script to verify installation and model functionality
"""

import torch
import numpy as np
from model import DiscoveryPipeline, BehaviorPreprocessor

def test_installation():
    """Test that all components can be imported and instantiated"""
    print("Testing installation...")

    # Test imports
    try:
        from model import DiscoveryPipeline, BehaviorPreprocessor
        from loss import discovery_loss
        from training import train_pipeline, FailureModeDetector
        from evaluation import IntrinsicEvaluator, ExtrinsicEvaluator
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    # Test model instantiation
    try:
        model = DiscoveryPipeline(
            input_dim=48,
            latent_dim=32,
            n_states=12,
            output_dim=121
        )
        print(f"✓ Model instantiated ({sum(p.numel() for p in model.parameters()):,} parameters)")
    except Exception as e:
        print(f"✗ Model instantiation error: {e}")
        return False

    # Test forward pass with dummy data
    try:
        dummy_data = torch.randn(2, 50, 48)  # 2 sequences, 50 frames, 48 features

        # Preprocess
        preprocessor = BehaviorPreprocessor()
        processed = preprocessor.preprocess(dummy_data)
        print(f"✓ Preprocessing works (output shape: {processed.shape})")

        # Forward pass
        model.eval()
        with torch.no_grad():
            reconstructed, codes = model(processed)
        print(f"✓ Forward pass works (recon: {reconstructed.shape}, codes: {codes.shape})")

        # Test generation
        synthetic = model.generate(n_samples=2, length=50)
        print(f"✓ Generation works (synthetic shape: {synthetic.shape})")

    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test loss computation
    try:
        from loss import discovery_loss
        loss, loss_dict = discovery_loss(processed, reconstructed, codes, model)
        print(f"✓ Loss computation works (total: {loss.item():.4f})")
        print(f"  - Reconstruction: {loss_dict['reconstruction']:.4f}")
        print(f"  - Codebook: {loss_dict['codebook']:.4f}")
        print(f"  - Temporal: {loss_dict['temporal']:.4f}")
    except Exception as e:
        print(f"✗ Loss computation error: {e}")
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Installation successful!")
    print("="*60)
    return True

if __name__ == '__main__':
    test_installation()
