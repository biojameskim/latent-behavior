"""Quick utility to inspect VQ-VAE training checkpoints (.pt)."""
"""
You can run this script from the command line like so:

    python training/inspect_checkpoint.py training/outputs/run_11_01_25_v5/checkpoint_epoch_240.pt
"""

import argparse
import pprint
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect a saved checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to the .pt file")
    parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Only list the top-level keys in the checkpoint and exit",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    print(f"\nLoaded checkpoint: {args.checkpoint}\n")
    print("Top-level keys:", list(ckpt.keys()))

    if args.show_keys:
        return

    if "epoch" in ckpt:
        print(f"Epoch: {ckpt['epoch']}")

    if "val_loss" in ckpt:
        print(f"Validation loss: {ckpt['val_loss']:.6f}")

    if "train_metrics" in ckpt:
        print("Train metrics:")
        pprint.pprint(ckpt["train_metrics"])

    if "args" in ckpt:
        print("\nSaved args:")
        pprint.pprint(ckpt["args"])

    state_dict = ckpt.get("model_state_dict")
    if state_dict is not None:
        n_params = sum(param.numel() for param in state_dict.values())
        print(f"\nModel state dict loaded with {len(state_dict)} tensors ({n_params:,} params)")


if __name__ == "__main__":
    main()
