#!/bin/bash
# Entry point: TINY model, fast dev/debug run.
#
# Use this to sanity-check the pipeline (dataset, degradation, model,
# training loop) quickly. NOT the reported model -- results from this
# config should not be used in the final write-up comparisons.
set -e
cd "$(dirname "$0")/.."

python3 train.py \
  --model_config configs/model_tiny.json \
  --degrade_config configs/degrade_realistic.json \
  --source_dir source_images \
  --splits_dir splits \
  --output_dir runs/tiny_dev \
  --epochs 5 \
  --batch_size 16 \
  --hr_patch_size 128 \
  --patches_per_image 8 \
  --num_workers 0 \
  "$@"
