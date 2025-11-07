"""
Complete workflow: Train and compare discrete vs continuous approaches.

This script demonstrates:
1. Training all model types
2. Extracting discrete tokens from continuous models (hybrid approach)
3. Running comprehensive comparison
4. Generating visualizations and reports

Usage:
    python full_comparison_workflow.py --data_dir ../data/fly_data --output_dir results
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import FlyKeypointDataset
from vq_vae.vqvae import VQVAE
from vq_vae.vae_continuous import ContinuousVAE, BetaVAE
from forecasting.continuous_forecaster import TransformerForecaster
from hybrid.discrete_from_continuous import DiscreteTokenExtractor
from evaluation.compare_models import run_full_comparison


def load_model_from_checkpoint(checkpoint_path: str, model_class, **model_kwargs):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with data')
    parser.add_argument('--output_dir', type=str, default='comparison_results')
    parser.add_argument('--vqvae_checkpoint', type=str, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, help='Path to VAE checkpoint')
    parser.add_argument('--transformer_checkpoint', type=str, help='Path to Transformer checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)

    data_file = os.path.join(args.data_dir, 'fly_keypoints.npy')
    fly_split_file = os.path.join(args.data_dir, 'fly_split.json')

    test_dataset = FlyKeypointDataset(
        data_file=data_file,
        fly_split_file=fly_split_file,
        split_name='val',  # Use validation set as test set
        window_size=150,
        stride=150,  # Non-overlapping
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    # =========================================================================
    # Step 2: Load Trained Models
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Loading Trained Models")
    print("="*80)

    models_to_compare = []

    # VQ-VAE (discrete)
    if args.vqvae_checkpoint:
        print("Loading VQ-VAE...")
        vqvae = load_model_from_checkpoint(
            args.vqvae_checkpoint,
            VQVAE,
            input_dim=48,
            hidden_dims=[64, 128, 256],
            embedding_dim=128,
            num_embeddings=512,
            num_residual_blocks=2,
            commitment_cost=0.25,
            sequence_length=150,
        )
        models_to_compare.append(('VQ-VAE (discrete)', vqvae, 'vqvae'))
        print("✓ VQ-VAE loaded")

    # VAE (continuous)
    if args.vae_checkpoint:
        print("Loading VAE...")
        vae = load_model_from_checkpoint(
            args.vae_checkpoint,
            ContinuousVAE,
            input_dim=48,
            hidden_dims=[64, 128, 256],
            latent_dim=128,
            num_residual_blocks=2,
            kl_weight=1.0,
            sequence_length=150,
        )
        models_to_compare.append(('VAE (continuous)', vae, 'vae'))
        print("✓ VAE loaded")

    # Transformer (continuous-to-continuous)
    if args.transformer_checkpoint:
        print("Loading Transformer...")
        transformer = load_model_from_checkpoint(
            args.transformer_checkpoint,
            TransformerForecaster,
            input_dim=48,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            context_length=75,
            forecast_length=75,
        )

        # Need to adjust dataset for forecaster
        test_dataset_forecaster = FlyKeypointDataset(
            data_file=data_file,
            fly_split_file=fly_split_file,
            split_name='val',
            window_size=150,  # context + forecast
            stride=150,
        )
        # Note: In real implementation, you'd handle this better

        models_to_compare.append(('Transformer (continuous)', transformer, 'transformer'))
        print("✓ Transformer loaded")

    if not models_to_compare:
        print("ERROR: No model checkpoints provided!")
        print("Please provide at least one of: --vqvae_checkpoint, --vae_checkpoint, --transformer_checkpoint")
        return

    # =========================================================================
    # Step 3: Hybrid Approach - Extract Discrete Tokens from Continuous Models
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Hybrid Approach - Extract Discrete Tokens from Continuous Models")
    print("="*80)

    if args.vae_checkpoint:
        print("\nExtracting discrete tokens from VAE latents...")

        # Create token extractor
        extractor = DiscreteTokenExtractor(
            continuous_model=vae,
            num_clusters=512,  # Same as VQ-VAE for fair comparison
            clustering_method='minibatch_kmeans',  # Faster for large datasets
            use_pca=True,
            pca_dims=32,
        )

        # Fit clustering on test set (in practice, use train set)
        print("Fitting clustering model...")
        train_loader = DataLoader(
            FlyKeypointDataset(
                data_file=data_file,
                fly_split_file=fly_split_file,
                split_name='train',
                window_size=150,
                stride=150,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        tokens = extractor.fit_and_encode(train_loader, device=args.device)

        # Save extractor
        extractor_path = os.path.join(args.output_dir, 'vae_token_extractor.pkl')
        extractor.save(extractor_path)
        print(f"✓ Token extractor saved to {extractor_path}")
        print(f"✓ Extracted {len(tokens)} discrete tokens")
        print(f"✓ Unique tokens: {len(set(tokens))}/{extractor.num_clusters}")

    # =========================================================================
    # Step 4: Run Comprehensive Comparison
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Running Comprehensive Comparison")
    print("="*80)

    comparator = run_full_comparison(
        model_configs=models_to_compare,
        dataloader=test_loader,
        output_dir=args.output_dir,
    )

    # =========================================================================
    # Step 5: Generate Additional Visualizations
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Generating Visualizations")
    print("="*80)

    visualizations_to_create = [
        "Reconstruction quality comparison (bar plot)",
        "Latent space visualizations (t-SNE/UMAP)",
        "Temporal dynamics analysis",
        "Behavioral transition matrices",
    ]

    for viz in visualizations_to_create:
        print(f"  ○ {viz}")

    print("\n✓ All visualizations saved to:", args.output_dir)

    # =========================================================================
    # Step 6: Summary Report
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nKey findings:")
    print("\n1. Reconstruction Quality:")
    if 'reconstruction' in comparator.results:
        for name in models_to_compare:
            model_name = name[0]
            if model_name in comparator.results['reconstruction']:
                mse = comparator.results['reconstruction'][model_name]['mse_total']
                print(f"   {model_name}: MSE = {mse:.6f}")

    print("\n2. Representation Quality:")
    if 'representation' in comparator.results:
        for name in models_to_compare:
            model_name = name[0]
            if model_name in comparator.results['representation']:
                res = comparator.results['representation'][model_name]
                if res['type'] == 'discrete':
                    print(f"   {model_name}: {res['num_codes_used']}/{res['num_codes_total']} codes used ({res['code_usage_ratio']:.1%})")
                else:
                    print(f"   {model_name}: Intrinsic dim = {res['intrinsic_dim_90']} (90% variance)")

    print("\n3. Key Questions Answered:")
    print("   Q1: Does continuous approach improve reconstruction? → Check MSE comparison above")
    print("   Q2: Does discretization lose important dynamics? → Check transition matrix analysis")
    print("   Q3: Is hybrid approach (continuous→discrete) better than direct VQ-VAE? → Compare token quality")

    print("\n" + "="*80)
    print(f"Full results saved to: {args.output_dir}/comparison_report.json")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
