#!/bin/bash
# Entry point: adversarial fine-tuning of the pure-L1 generator.
#
# Warm-starts from runs/small_realistic/checkpoints/best.pt (must exist --
# run scripts/run_final_realistic.sh first) and fine-tunes with a combined
# L1 + adversarial loss against a PatchGAN discriminator, following SRGAN's
# own pretrain-then-fine-tune recipe for training stability.
#
# HYPERPARAMETER HISTORY (see WRITEUP / deep-dive report, Section 6.4):
#   V3a (first attempt, NOT used here): lr_d=1e-4, w_adv=0.01
#     -> discriminator overpowered the generator within a few epochs
#        (disc_loss collapsed to 0.07-0.09, gen_adv plateaued ~0.75-0.78,
#        no visible output change from the pure-L1 checkpoint).
#   V3b (rebalanced, THESE ARE THE DEFAULTS BELOW): lr_d=2e-5, w_adv=0.15
#     -> disc_loss stabilized in a healthy oscillating 0.20-0.28 band,
#        gen_l1 settled at a new stable equilibrium (~0.06 vs ~0.048 for
#        pure L1), modest but genuine qualitative texture improvement.
#
# Expect PSNR/SSIM to hold steady or dip slightly relative to the pure-L1
# checkpoint -- judge success primarily by qualitative sharpness, not these
# metrics (see train_gan.py's closing note).
set -e
cd "$(dirname "$0")/.."

python3 train_gan.py \
  --generator_checkpoint runs/small_realistic/checkpoints/best.pt \
  --degrade_config configs/degrade_realistic.json \
  --source_dir source_images \
  --splits_dir splits \
  --output_dir runs/small_gan \
  --epochs 20 \
  --batch_size 32 \
  --lr_g 1e-5 \
  --lr_d 2e-5 \
  --w_l1 1.0 \
  --w_adv 0.15 \
  --hr_patch_size 128 \
  --patches_per_image 16 \
  "$@"