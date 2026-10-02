#!/bin/bash
# Entry point: SMALL model + REALISTIC degradation.
#
# This is the main reported / final model: randomized blur, noise, JPEG
# compression, and occasional repeated resave, intended to generalize beyond
# the specific degradation script used to generate its own training pairs.
set -e
cd "$(dirname "$0")/.."

python3 train.py \
  --model_config configs/model_small.json \
  --degrade_config configs/degrade_realistic.json \
  --source_dir source_images \
  --splits_dir splits \
  --output_dir runs/small_realistic \
  --epochs 40 \
  --batch_size 32 \
  --hr_patch_size 128 \
  --patches_per_image 16 \
  "$@"
