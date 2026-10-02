"""
infer.py

Standalone demo/inference script: takes ONE image, produces a synthetic
low-res version (using the same degradation pipeline as training), runs it
through a trained model, and saves a clean side-by-side comparison image at
full resolution -- intended for demos/presentations, not just metrics.

Because the model is fully convolutional (see models/sr_net.py), it isn't
restricted to the patch size it was trained on -- it can run on a full image
of arbitrary size at inference time, which is what this script does (unlike
train.py/evaluate.py, which operate on fixed-size patches for efficient
batched training/evaluation).

Usage:
    python infer.py \
        --checkpoint runs/small_realistic/checkpoints/best.pt \
        --image source_images/108073.jpg \
        --degrade_config configs/degrade_realistic.json \
        --output_dir demo_output

Produces demo_output/<image_stem>_comparison.png with three panels:
    [ synthetic low-res input (upscaled with nearest-neighbor so you can see
      the actual pixels, not smoothed) | model super-resolved output |
      original ground truth ]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "data"))
sys.path.insert(0, str(Path(__file__).parent / "models"))

from degrade import build_degrader
from sr_net import SRNet


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = np.asarray(img.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).astype("uint8"))


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
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


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Adds a small readable label bar under an image panel."""
    bar_h = 28
    labeled = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    labeled.paste(img, (0, 0))
    draw = ImageDraw.Draw(labeled)
    draw.text((8, img.height + 6), text, fill=(255, 255, 255))
    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True,
                         help="Path to a HIGH-res source image; a synthetic "
                              "low-res version will be generated from it")
    parser.add_argument("--degrade_config", type=str,
                         default="configs/degrade_realistic.json")
    parser.add_argument("--output_dir", type=str, default="demo_output")
    parser.add_argument("--seed", type=int, default=123,
                         help="Fixes the degradation so results are reproducible")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, model_cfg = load_model(args.checkpoint, device)
    scale_factor = model_cfg["scale_factor"]
    print(f"Loaded model: {model_cfg['name']} (scale={scale_factor})")

    hr_img = Image.open(args.image).convert("RGB")
    # Crop to a multiple of scale_factor so the LR/HR sizes divide evenly.
    w, h = hr_img.size
    w = (w // scale_factor) * scale_factor
    h = (h // scale_factor) * scale_factor
    hr_img = hr_img.crop((0, 0, w, h))

    degrader = build_degrader(args.degrade_config, seed=args.seed)
    lr_img = degrader(hr_img)
    print(f"HR size: {hr_img.size}, synthetic LR size: {lr_img.size}")

    lr_tensor = to_tensor(lr_img).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_img = to_pil(sr_tensor)

    # Nearest-neighbor upscale of the LR image just for DISPLAY purposes, so
    # you can see the actual low-res pixels/artifacts rather than having them
    # smoothed away -- this is NOT what the model sees (the model sees the
    # small LR image directly).
    lr_display = lr_img.resize((w, h), Image.NEAREST)

    panels = [
        add_label(lr_display, f"Low-res input ({lr_img.size[0]}x{lr_img.size[1]}, shown at {w}x{h})"),
        add_label(sr_img, f"Model output ({scale_factor}x super-resolved)"),
        add_label(hr_img, "Original ground truth"),
    ]

    gap = 10
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    total_h = max(p.height for p in panels)
    comparison = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    x = 0
    for p in panels:
        comparison.paste(p, (x, 0))
        x += p.width + gap

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    out_path = out_dir / f"{stem}_comparison.png"
    comparison.save(out_path)
    print(f"Saved comparison -> {out_path}")


if __name__ == "__main__":
    main()