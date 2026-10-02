#!/bin/bash
# Entry point: full evaluation / comparison.
#
# Evaluates the bicubic baseline, the naive-degradation-trained model, and
# the realistic-degradation-trained model, ALL against a common test set
# built with the REALISTIC degradation config (the harder, more real-world-
# like condition). This produces the metrics table and qualitative grid
# referenced throughout the write-up's "Evaluation and Analysis" section.
#
# Run scripts/run_baseline_naive.sh and scripts/run_final_realistic.sh first
# so both checkpoints exist.
set -e
cd "$(dirname "$0")/.."

python3 evaluate.py \
  --source_dir source_images \
  --splits_dir splits \
  --test_degrade_config configs/degrade_realistic.json \
  --checkpoints runs/small_naive/checkpoints/best.pt runs/small_realistic/checkpoints/best.pt \
  --labels naive_trained realistic_trained \
  --output_dir eval_results \
  "$@"
