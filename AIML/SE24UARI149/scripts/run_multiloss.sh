#!/bin/bash
# Entry point: SMALL model + REALISTIC degradation + MULTI-COMPONENT LOSS.
#
# Same architecture and degradation as scripts/run_final_realistic.sh, but
# trained with L1 + gradient/edge loss + SSIM loss instead of pure L1.
#
# Motivation: the pure-L1 model (runs/small_realistic) converged (flat
# val_loss/val_psnr across epochs 30-39) at only a modest improvement over
# bicubic baseline, with output visibly softer than ground truth -- the
# expected signature of pure pixel-loss training under super-resolution's
# ill-posedness (many plausible HR outputs per LR input -> L1 converges
# toward a "safe average" rather than a sharp, committed reconstruction).
# The gradient loss term directly penalizes edge over-smoothing relative to
# ground truth; the SSIM term rewards preserved local structure/contrast.
# Neither uses any pretrained/external network -- both are computed via
# plain tensor arithmetic on the model's own output.
set -e
cd "$(dirname "$0")/.."

python3 train.py \
  --model_config configs/model_small_multiloss.json \
  --degrade_config configs/degrade_realistic.json \
  --source_dir source_images \
  --splits_dir splits \
  --output_dir runs/small_multiloss \
  --epochs 40 \
  --batch_size 32 \
  --hr_patch_size 128 \
  --patches_per_image 16 \
  "$@"