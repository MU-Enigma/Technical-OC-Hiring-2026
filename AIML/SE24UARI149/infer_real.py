"""
infer_real.py

Like infer.py, but for testing on a GENUINELY low-resolution image (an old
photo, a screenshot, a compressed web image) rather than a synthetic LR
image generated from a high-res source.

This is the stronger, more honest generalization test: infer.py measures
"can the model undo my own synthetic degradation pipeline," which is
somewhat self-fulfilling since the model trained on exactly that pipeline.
This script measures "can the model handle a real low-res image whose
actual degradation history is unknown and was never controlled by this
project's degradation pipeline at all" -- the real target claim of the
whole assignment.

There is no ground truth here (if you had the original high-res image, it
wouldn't be a real low-res image in the first place), so this only produces
a two-panel comparison: [real low-res input | model output], both shown at
the same display size for direct visual comparison.

Usage:
    python infer_real.py \
        --checkpoint runs/small_realistic/checkpoints/best.pt \
        --image path/to/real_low_res_photo.jpg \
        --output_dir demo_output_real
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent / "models"))
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
                         help="Path to a REAL low-res image (no synthetic "
                              "degradation is applied -- this image is fed "
                              "to the model as-is)")
    parser.add_argument("--output_dir", type=str, default="demo_output_real")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, model_cfg = load_model(args.checkpoint, device)
    scale_factor = model_cfg["scale_factor"]
    print(f"Loaded model: {model_cfg['name']} (scale={scale_factor})")

    lr_img = Image.open(args.image).convert("RGB")
    w, h = lr_img.size
    print(f"Real low-res input size: {w}x{h}")

    # The model is fully convolutional (see models/sr_net.py), so it accepts
    # any input size directly -- no cropping to a multiple of scale_factor
    # is strictly required here since we aren't comparing against a HR
    # ground truth that needs matching dimensions. We still do it for
    # cleanliness / to avoid PixelShuffle producing a non-integer-friendly
    # output size on an odd input.
    w_crop = (w // scale_factor) * scale_factor
    h_crop = (h // scale_factor) * scale_factor
    if (w_crop, h_crop) != (w, h):
        lr_img = lr_img.crop((0, 0, w_crop, h_crop))
        print(f"Cropped to {w_crop}x{h_crop} (multiple of scale_factor={scale_factor})")

    lr_tensor = to_tensor(lr_img).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_img = to_pil(sr_tensor)

    out_w, out_h = sr_img.size
    # Nearest-neighbor upscale of the real LR input for DISPLAY only, so you
    # can see it at the same size as the output without smoothing away its
    # actual pixels/artifacts. The model itself only ever saw the small
    # original image, not this upscaled display version.
    lr_display = lr_img.resize((out_w, out_h), Image.NEAREST)

    panels = [
        add_label(lr_display, f"Real low-res input ({w_crop}x{h_crop}, shown at {out_w}x{out_h})"),
        add_label(sr_img, f"Model output ({scale_factor}x super-resolved, {out_w}x{out_h})"),
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
    out_path = out_dir / f"{stem}_real_comparison.png"
    comparison.save(out_path)
    print(f"Saved comparison -> {out_path}")


if __name__ == "__main__":
    main()