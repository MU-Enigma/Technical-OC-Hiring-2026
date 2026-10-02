"""
evaluate.py
-----------
1. Quantitative comparison table (PSNR/SSIM) across bicubic / SRCNN /
   SRResNet-lite(bicubic-only) / SRResNet-lite(full degradation), on the
   held-out SYNTHETIC test split.
2. Qualitative comparison on a small folder of REAL low-quality images
   with no ground truth - the strongest generalization evidence.

Usage:
    python -m src.evaluate --data_dir data/images --real_dir data/real_lowres \
        --checkpoints_root outputs --scale 4
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import list_images, split_files, SRDataset
from src.models import build_model
from src.utils import get_device, psnr, ssim, save_comparison_grid, tensor_to_pil
from src.train import call_model, evaluate_loader


MODEL_REGISTRY = {
    "bicubic":                       ("bicubic", None),
    "srcnn":                         ("srcnn", "srcnn"),
    "srresnet_bicubic_only":         ("srresnet", "srresnet_bicubic"),
    "srresnet_full_degradation":     ("srresnet", "srresnet_full"),
}


def load_trained_model(arch_name, run_name, checkpoints_root, scale, device):
    model = build_model(arch_name, scale=scale).to(device)
    if arch_name == "bicubic":
        return model
    ckpt_path = os.path.join(checkpoints_root, run_name, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        print(f"[evaluate] WARNING: no checkpoint at {ckpt_path}, skipping {run_name}")
        return None
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def quantitative_comparison(data_dir, checkpoints_root, scale, hr_crop, batch_size, device):
    files = list_images(data_dir)
    _, _, test_files = split_files(files)
    test_ds = SRDataset(test_files, scale=scale, hr_crop=hr_crop, strength="medium", train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    for label, (arch, run_name) in MODEL_REGISTRY.items():
        model = load_trained_model(arch, run_name, checkpoints_root, scale, device)
        if model is None:
            continue
        metrics = evaluate_loader(model, arch, test_loader, device)
        results[label] = metrics
        print(f"[evaluate] {label:30s} PSNR={metrics['psnr']:.3f}  SSIM={metrics['ssim']:.4f}")

    return results


def qualitative_on_real_photos(real_dir, checkpoints_root, scale, device, out_path):
    real_files = list_images(real_dir)
    if len(real_files) == 0:
        print(f"[evaluate] no images found in {real_dir}, skipping qualitative real-photo eval")
        return

    models = {}
    for label, (arch, run_name) in MODEL_REGISTRY.items():
        m = load_trained_model(arch, run_name, checkpoints_root, scale, device)
        if m is not None:
            models[label] = (arch, m)

    labels = list(models.keys())
    n_cols = 1 + len(labels)
    thumb = 200
    pad = 6
    grid = Image.new("RGB", (thumb * n_cols + pad * (n_cols + 1),
                              (thumb + pad) * len(real_files) + pad),
                      color=(15, 15, 15))

    for row, path in enumerate(real_files):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        target_small = 128
        ratio = target_small / min(w, h)
        lr_small_img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BICUBIC)
        lr_bicubic_img = lr_small_img.resize(
            (lr_small_img.width * scale, lr_small_img.height * scale), Image.BICUBIC
        )

        lr_small_t = TF.to_tensor(lr_small_img).unsqueeze(0).to(device)
        lr_bicubic_t = TF.to_tensor(lr_bicubic_img).unsqueeze(0).to(device)

        y = pad + row * (thumb + pad)
        grid.paste(lr_small_img.resize((thumb, thumb)), (pad, y))

        for col, label in enumerate(labels, start=1):
            arch, model = models[label]
            with torch.no_grad():
                pred = call_model(model, arch, lr_small_t, lr_bicubic_t).clamp(0, 1)
            pred_img = tensor_to_pil(pred[0]).resize((thumb, thumb))
            x = pad + col * (thumb + pad)
            grid.paste(pred_img, (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"[evaluate] saved real-photo qualitative grid: {out_path}  "
          f"(columns: input, {', '.join(labels)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--real_dir", default=None)
    parser.add_argument("--checkpoints_root", default="outputs")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--hr_crop", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", default="outputs/eval")
    args = parser.parse_args()

    device = get_device()
    print(f"[evaluate] device = {device}")

    results = quantitative_comparison(args.data_dir, args.checkpoints_root,
                                       args.scale, args.hr_crop, args.batch_size, device)

    os.makedirs(args.out_dir, exist_ok=True)
    table_path = os.path.join(args.out_dir, "comparison_table.json")
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] saved comparison table: {table_path}")

    md_path = os.path.join(args.out_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("| Model | PSNR | SSIM |\n|---|---|---|\n")
        for label, m in results.items():
            f.write(f"| {label} | {m['psnr']:.3f} | {m['ssim']:.4f} |\n")
    print(f"[evaluate] saved markdown table: {md_path}")

    if args.real_dir:
        qualitative_on_real_photos(args.real_dir, args.checkpoints_root, args.scale,
                                    device, os.path.join(args.out_dir, "real_photo_comparison.png"))


if __name__ == "__main__":
    main()