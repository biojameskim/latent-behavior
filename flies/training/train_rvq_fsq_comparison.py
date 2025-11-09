"""
RVQ and FSQ Focused Comparison Script

Based on initial results showing RVQ (loss: 4.74) and FSQ (loss: 14.17)
significantly outperform other methods. This script explores variations
of these two promising approaches.

Usage:
    python train_rvq_fsq_comparison.py --train_data /path/to/data.npy \
        --fly_split_file /path/to/split.pkl \
        --methods rvq_base fsq_base rvq_deep rvq_large fsq_small \
        --epochs 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_dataloaders
from vq_vae.vqvae_unified import UnifiedVQVAE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


# Configuration dictionary for RVQ and FSQ variations
QUANTIZER_CONFIGS = {
    # ============ RVQ VARIATIONS ============

    'rvq_base': {
        'method': 'rvq',
        'kwargs': {
            'num_quantizers': 4,
            'kmeans_init': True,
            'threshold_ema_dead_code': 2,
            'shared_codebook': False
        },
        'codebook_size_override': None,  # Uses --num_embeddings
        'description': 'Baseline RVQ (4 quantizers × 32 codes, best from initial test)'
    },

    'rvq_deep': {
        'method': 'rvq',
        'kwargs': {
            'num_quantizers': 8,  # More hierarchy
            'kmeans_init': True,
            'threshold_ema_dead_code': 2,
            'shared_codebook': False
        },
        'codebook_size_override': None,
        'description': 'Deep RVQ (8 quantizers for finer refinement)'
    },

    'rvq_large': {
        'method': 'rvq',
        'kwargs': {
            'num_quantizers': 4,
            'kmeans_init': True,
            'threshold_ema_dead_code': 2,
            'shared_codebook': False
        },
        'codebook_size_override': 64,  # Larger codebook per quantizer
        'description': 'Large RVQ (4 quantizers × 64 codes for more capacity)'
    },

    'rvq_xlarge': {
        'method': 'rvq',
        'kwargs': {
            'num_quantizers': 6,
            'kmeans_init': True,
            'threshold_ema_dead_code': 2,
            'shared_codebook': False
        },
        'codebook_size_override': 64,
        'description': 'XLarge RVQ (6 quantizers × 64 codes, maximum capacity)'
    },

    'rvq_shared': {
        'method': 'rvq',
        'kwargs': {
            'num_quantizers': 4,
            'kmeans_init': True,
            'threshold_ema_dead_code': 2,
            'shared_codebook': True  # Share codebook across quantizers
        },
        'codebook_size_override': 64,
        'description': 'Shared RVQ (4 quantizers sharing 64-code codebook, parameter efficient)'
    },

    'rvq_shallow_large': {
        'method': 'rvq',
        'kwargs': {
            'num_quantizers': 2,
            'kmeans_init': True,
            'threshold_ema_dead_code': 2,
            'shared_codebook': False
        },
        'codebook_size_override': 128,
        'description': 'Shallow-Large RVQ (2 quantizers × 128 codes, faster alternative)'
    },

    # ============ FSQ VARIATIONS ============

    'fsq_base': {
        'method': 'fsq',
        'kwargs': {
            'levels': [8, 5, 5, 5]  # 1000 codes
        },
        'codebook_size_override': None,
        'description': 'Baseline FSQ (1000 codes, best from initial test)'
    },

    'fsq_small': {
        'method': 'fsq',
        'kwargs': {
            'levels': [7, 5, 5, 3]  # 525 codes
        },
        'codebook_size_override': None,
        'description': 'Small FSQ (525 codes, more efficient)'
    },

    'fsq_large': {
        'method': 'fsq',
        'kwargs': {
            'levels': [8, 6, 5, 5, 5]  # 6000 codes
        },
        'codebook_size_override': None,
        'description': 'Large FSQ (6000 codes, maximum expressiveness)'
    },

    'fsq_balanced': {
        'method': 'fsq',
        'kwargs': {
            'levels': [8, 8, 8, 5]  # 2560 codes
        },
        'codebook_size_override': None,
        'description': 'Balanced FSQ (2560 codes, balanced dimensions)'
    },

    'fsq_minimal': {
        'method': 'fsq',
        'kwargs': {
            'levels': [6, 5, 5, 3]  # 450 codes
        },
        'codebook_size_override': None,
        'description': 'Minimal FSQ (450 codes, most efficient)'
    },

    'fsq_xlarge': {
        'method': 'fsq',
        'kwargs': {
            'levels': [9, 8, 7, 6, 5]  # 15120 codes
        },
        'codebook_size_override': None,
        'description': 'XLarge FSQ (15120 codes, very high capacity)'
    },
}


def _load_fly_id_list(entries):
    fly_ids = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Fly filter entries must be dictionaries")
        fly_ids.add((str(item['sequence_id']), int(item['fly_idx'])))
    return fly_ids


def load_fly_filters(fly_split_file=None, train_split_name='train',
                    val_split_name='val', train_filter_path=None,
                    val_filter_path=None):
    train_ids = None
    val_ids = None

    if fly_split_file:
        with open(fly_split_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        if train_split_name in content:
            train_ids = _load_fly_id_list(content[train_split_name])
        if val_split_name in content:
            val_ids = _load_fly_id_list(content[val_split_name])

    if train_filter_path:
        with open(train_filter_path, 'r', encoding='utf-8') as f:
            train_ids = _load_fly_id_list(json.load(f))
    if val_filter_path:
        with open(val_filter_path, 'r', encoding='utf-8') as f:
            val_ids = _load_fly_id_list(json.load(f))

    return train_ids, val_ids


def train_epoch(model, train_loader, optimizer, device, epoch, method_name, grad_clip_norm=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    for batch_idx, x in enumerate(train_loader):
        x = x.to(device)

        # Forward pass
        x_recon, vq_loss, perplexity, _, _ = model(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Total loss
        loss = recon_loss + vq_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'perplexity': total_perplexity / n_batches
    }


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            # Forward pass
            x_recon, vq_loss, perplexity, _, _ = model(x)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)

            # Total loss
            loss = recon_loss + vq_loss

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches,
        'perplexity': total_perplexity / n_batches
    }


def train_single_method(method_name, args, train_loader, val_loader, device):
    """Train a single quantization method."""
    LOG.info("")
    LOG.info("="*70)
    LOG.info(f"Training method: {method_name.upper()}")
    LOG.info("="*70)

    # Get quantizer config
    if method_name not in QUANTIZER_CONFIGS:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {list(QUANTIZER_CONFIGS.keys())}")

    config = QUANTIZER_CONFIGS[method_name]
    LOG.info(f"Description: {config['description']}")
    LOG.info(f"Quantizer config: {config}")

    # Use method-specific codebook size if specified, otherwise use command line arg
    num_embeddings = config.get('codebook_size_override')
    if num_embeddings is not None:
        LOG.info(f"Using method-specific codebook size: {num_embeddings} (override)")
    else:
        num_embeddings = args.num_embeddings
        LOG.info(f"Using command-line codebook size: {num_embeddings}")

    # Create model
    model = UnifiedVQVAE(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        num_embeddings=num_embeddings,
        sequence_length=args.window_size,
        num_residual_blocks=args.num_residual_blocks,
        commitment_cost=args.commitment_cost,
        quantizer_method=config['method'],
        quantizer_kwargs=config['kwargs']
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f"Model has {n_params:,} trainable parameters")
    LOG.info(f"Codebook size: {model.quantizer.get_codebook_size()}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    # Learning rate scheduler
    t_max = args.lr_scheduler_tmax or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # Create method-specific output directory
    method_output_dir = Path(args.output_dir) / method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = defaultdict(list)
    best_val_loss = float('inf')

    # Training loop
    LOG.info(f"\n[{method_name}] Starting training for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            method_name, grad_clip_norm=args.grad_clip_norm
        )

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device)
        else:
            val_metrics = {'loss': 0, 'recon_loss': 0, 'vq_loss': 0, 'perplexity': 0}

        # Update learning rate
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_vq_loss'].append(train_metrics['vq_loss'])
        history['train_perplexity'].append(train_metrics['perplexity'])

        if val_loader is not None:
            history['val_loss'].append(val_metrics['loss'])
            history['val_recon_loss'].append(val_metrics['recon_loss'])
            history['val_vq_loss'].append(val_metrics['vq_loss'])
            history['val_perplexity'].append(val_metrics['perplexity'])

        # Print progress
        if val_loader is not None:
            LOG.info(
                f"[{method_name}] Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Perplexity: {val_metrics['perplexity']:.2f}"
            )
        else:
            LOG.info(
                f"[{method_name}] Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Perplexity: {train_metrics['perplexity']:.2f}"
            )

        # Save best model
        if val_loader is not None and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = method_output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'method': method_name,
                'config': config,
                'args': vars(args)
            }, best_model_path)

        # Save checkpoint periodically
        if epoch % args.save_every == 0:
            checkpoint_path = method_output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'method': method_name,
                'config': config,
                'args': vars(args)
            }, checkpoint_path)

    # Save final model and history
    final_path = method_output_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'method': method_name,
        'config': config,
        'args': vars(args)
    }, final_path)

    # Save training history as JSON
    history_path = method_output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    LOG.info(f"[{method_name}] Training complete!")
    LOG.info(f"[{method_name}] Best val loss: {best_val_loss:.4f}")

    return history, best_val_loss


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    LOG.info("Creating dataloaders...")
    train_fly_ids, val_fly_ids = load_fly_filters(
        fly_split_file=args.fly_split_file,
        train_split_name=args.train_split_name,
        val_split_name=args.val_split_name,
        train_filter_path=args.train_fly_filter,
        val_filter_path=args.val_fly_filter,
    )

    if args.val_data or val_fly_ids is not None:
        train_loader, val_loader = create_dataloaders(
            train_data_file=args.train_data,
            val_data_file=args.val_data,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_fly_ids=train_fly_ids,
            val_fly_ids=val_fly_ids
        )
    else:
        train_loader = create_dataloaders(
            train_data_file=args.train_data,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_fly_ids=train_fly_ids
        )
        val_loader = None

    LOG.info(f"Train batches: {len(train_loader)}")
    if val_loader:
        LOG.info(f"Val batches: {len(val_loader)}")

    # Print overview of methods being trained
    LOG.info("\n" + "="*70)
    LOG.info("METHODS TO TRAIN:")
    LOG.info("="*70)
    for method in args.methods:
        if method in QUANTIZER_CONFIGS:
            desc = QUANTIZER_CONFIGS[method]['description']
            LOG.info(f"  {method:20s} - {desc}")
        else:
            LOG.warning(f"  {method:20s} - UNKNOWN METHOD (will skip)")
    LOG.info("="*70)

    # Train each method
    results = {}
    for method in args.methods:
        try:
            history, best_val_loss = train_single_method(
                method, args, train_loader, val_loader, device
            )
            results[method] = {
                'history': history,
                'best_val_loss': best_val_loss
            }
        except Exception as e:
            LOG.error(f"Failed to train {method}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save comparison summary
    LOG.info("\n" + "="*70)
    LOG.info("COMPARISON SUMMARY")
    LOG.info("="*70)

    summary = {}
    for method, result in results.items():
        best_val = result['best_val_loss']
        final_train = result['history']['train_loss'][-1] if result['history']['train_loss'] else float('inf')
        final_perp = result['history']['val_perplexity'][-1] if result['history']['val_perplexity'] else 0

        summary[method] = {
            'best_val_loss': best_val,
            'final_train_loss': final_train,
            'final_perplexity': final_perp,
            'description': QUANTIZER_CONFIGS[method]['description']
        }
        LOG.info(f"{method:20s} | Val: {best_val:.4f} | Train: {final_train:.4f} | Perp: {final_perp:.2f}")

    summary_path = output_dir / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    LOG.info(f"\nComparison complete! Results saved to: {output_dir}")
    LOG.info(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare RVQ and FSQ variations')

    # Methods to compare
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['rvq_base', 'rvq_deep', 'rvq_large', 'fsq_base', 'fsq_small', 'fsq_large'],
                       help='Methods to compare (see QUANTIZER_CONFIGS for options)')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--window_size', type=int, default=150)
    parser.add_argument('--stride', type=int, default=150)

    # Model arguments
    parser.add_argument('--input_dim', type=int, default=48)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--num_residual_blocks', type=int, default=2)
    parser.add_argument('--commitment_cost', type=float, default=0.25)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--lr_scheduler_tmax', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip_norm', type=float, default=None)

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/rvq_fsq_comparison')
    parser.add_argument('--save_every', type=int, default=10)

    # Fly-level split arguments
    parser.add_argument('--fly_split_file', type=str, default=None)
    parser.add_argument('--train_split_name', type=str, default='train')
    parser.add_argument('--val_split_name', type=str, default='val')
    parser.add_argument('--train_fly_filter', type=str, default=None)
    parser.add_argument('--val_fly_filter', type=str, default=None)

    args = parser.parse_args()

    LOG.info("Arguments:")
    for arg, value in vars(args).items():
        LOG.info(f"  {arg}: {value}")

    main(args)
