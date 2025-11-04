#!/bin/bash

VERSIONS=("v1" "v2" "v3" "v4" "v5")
for VERSION in "${VERSIONS[@]}"; do
    echo "Inspecting checkpoint (epoch 250) for version: $VERSION"
    python training/inspect_checkpoint.py training/outputs/run_11_01_25_"$VERSION"/checkpoint_epoch_250.pt >> training/compare_results.txt
    echo "Results written to training/compare_results.txt"
done
