"""
evaluate.py

Evaluates one or more trained models (plus a zero-parameter bicubic baseline)
on the test split, and produces:

  1. A metrics table (PSNR/SSIM) written to CSV and printed to console.
  2. Qualitative side-by-side image grids for a handful of test images,
     comparing bicubic baseline vs. each provided model vs. ground truth.

This script is deliberately designed to answer the "beyond loss values" and
"baseline vs final model" requirements directly: run it once with just a
bicubic baseline for reference, and again passing --checkpoints for each
trained model (e.g. naive-degradation-trained vs realistic-degradation-trained)
to get a genuine apples-to-apples comparison table.

Usage:
    python evaluate.py \
        --source_dir source_images --splits_dir splits \
        --test_degrade_config configs/degrade_realistic.json \
        --checkpoints runs/small_naive/checkpoints/best.pt runs/small_realistic/checkpoints/best.pt \
        --labels naive_trained realistic_trained \
        --output_dir eval_results

Important evaluation-validity note: all models (and the bicubic baseline) are
evaluated on the SAME test-split LR/HR pairs, generated with a SINGLE fixed
--test_degrade_config. This is what makes the naive-vs-realistic comparison
fair: we are asking "how well does each *trained* model perform against a
common, realistic test condition," not letting each model be evaluated on
whatever degradation it happened to train on.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid

sys.path.insert(0, str(Path(__file__).parent / "data"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from dataset import SRPatchDataset
from sr_net import SRNet
from metrics import compute_metrics
from checkpoint import load_checkpoint


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_from_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = SRNet(
        num_blocks=model_cfg["num_blocks"],
        num_channels=model_cfg["num_channels"],
        scale_factor=model_cfg["scale_factor"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_cfg


@torch.no_grad()
def bicubic_predict(lr_batch: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.clamp(
        F.interpolate(lr_batch, scale_factor=scale_factor, mode="bicubic",
                      align_corners=False),
        0.0, 1.0,
    )


@torch.no_grad()
def evaluate_predictor(predict_fn, loader, device) -> dict:
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = predict_fn(lr)
        m = compute_metrics(pred, hr)
        total_psnr += m["psnr"]
        total_ssim += m["ssim"]
        n += 1
    return {"psnr": total_psnr / max(1, n), "ssim": total_ssim / max(1, n)}


def build_qualitative_grid(lr, hr, predictions: dict, scale_factor: int, out_path: str,
                            max_samples: int = 4):
    """predictions: dict label -> prediction tensor (B,C,H,W), already computed."""
    n = min(max_samples, lr.shape[0])
    bicubic = bicubic_predict(lr[:n], scale_factor)
    hr_n = hr[:n]

    labels = ["bicubic"] + list(predictions.keys()) + ["ground_truth"]
    rows = []
    for i in range(n):
        rows.append(bicubic[i])
        for label in predictions:
            rows.append(torch.clamp(predictions[label][:n][i], 0.0, 1.0))
        rows.append(hr_n[i])

    grid_tensor = torch.stack(rows, dim=0)
    n_cols = len(labels)
    grid = make_grid(grid_tensor, nrow=n_cols, padding=4, pad_value=1.0)
    grid_img = Image.fromarray(
        (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_path)
    print(f"Saved qualitative grid ({', '.join(labels)}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="source_images")
    parser.add_argument("--splits_dir", type=str, default="splits")
    parser.add_argument("--test_degrade_config", type=str, required=True,
                         help="Degradation config used to build the common "
                              "test set all models/baseline are evaluated against")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=[],
                         help="Paths to trained model checkpoint(s)")
    parser.add_argument("--labels", type=str, nargs="*", default=[],
                         help="Display labels for each checkpoint, same order")
    parser.add_argument("--hr_patch_size", type=int, default=128)
    parser.add_argument("--patches_per_image", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.checkpoints):
        raise ValueError("--labels must match --checkpoints in count if provided")
    labels = args.labels or [Path(c).parent.parent.name for c in args.checkpoints]

    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = SRPatchDataset(
        source_dir=args.source_dir,
        file_list_path=str(Path(args.splits_dir) / "test.txt"),
        degrade_config_path=args.test_degrade_config,
        scale_factor=args.scale_factor,
        hr_patch_size=args.hr_patch_size,
        patches_per_image=args.patches_per_image,
        split="test",
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_ds)} "
          f"(degraded with {args.test_degrade_config})")

    results = {}

    # 1. Zero-parameter bicubic baseline.
    bicubic_metrics = evaluate_predictor(
        lambda lr: bicubic_predict(lr, args.scale_factor), test_loader, device
    )
    results["bicubic_baseline"] = bicubic_metrics
    print(f"bicubic_baseline: PSNR={bicubic_metrics['psnr']:.3f} "
          f"SSIM={bicubic_metrics['ssim']:.4f}")

    # 2. Each trained model checkpoint.
    models = {}
    for ckpt_path, label in zip(args.checkpoints, labels):
        model, model_cfg = load_model_from_checkpoint(ckpt_path, device)
        models[label] = model
        m = evaluate_predictor(lambda lr, model=model: model(lr), test_loader, device)
        results[label] = m
        print(f"{label}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f}")

    # Write metrics table.
    csv_path = output_dir / "metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "psnr", "ssim"])
        for name, m in results.items():
            writer.writerow([name, f"{m['psnr']:.4f}", f"{m['ssim']:.5f}"])
    print(f"Metrics table -> {csv_path}")

    with open(output_dir / "metrics_table.json", "w") as f:
        json.dump(results, f, indent=2)

    # Qualitative grid: one batch of real test samples, run through each model.
    lr_batch, hr_batch = next(iter(test_loader))
    lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
    predictions = {}
    with torch.no_grad():
        for label, model in models.items():
            predictions[label] = model(lr_batch)

    build_qualitative_grid(
        lr_batch, hr_batch, predictions, args.scale_factor,
        str(output_dir / "qualitative_comparison.png"),
    )

    print(f"\nAll evaluation outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
