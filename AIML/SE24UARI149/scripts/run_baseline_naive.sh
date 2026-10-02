#!/bin/bash
# Entry point: SMALL model + NAIVE degradation.
#
# This is the "too clean" baseline referenced in the write-up: bicubic-only
# downsampling with no blur/noise/JPEG. Trained to demonstrate the domain-gap
# problem when evaluated against realistically-degraded test data
# (see evaluate.py / scripts/run_evaluation.sh).
set -e
cd "$(dirname "$0")/.."

python3 train.py \
  --model_config configs/model_small.json \
  --degrade_config configs/degrade_naive.json \
  --source_dir source_images \
  --splits_dir splits \
  --output_dir runs/small_naive \
  --epochs 40 \
  --batch_size 32 \
  --hr_patch_size 128 \
  --patches_per_image 16 \
  "$@"
