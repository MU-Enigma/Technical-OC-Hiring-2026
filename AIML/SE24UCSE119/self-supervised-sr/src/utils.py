"""
utils.py
--------
Shared helpers: reproducibility, device selection, metrics (PSNR/SSIM),
and saving side-by-side comparison grids for training progression images.
"""

import random
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(pred, target, max_val=1.0):
    """pred, target: (B, 3, H, W) tensors in [0, 1]. Returns mean PSNR over batch."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr_vals = 10 * torch.log10((max_val ** 2) / mse)
    return psnr_vals.mean().item()


def ssim(pred, target):
    """pred, target: (B, 3, H, W) tensors in [0, 1]. Returns mean SSIM over batch,
    computed per-image on CPU via skimage (channel_axis for RGB)."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    scores = []
    for p, t in zip(pred_np, target_np):
        p = np.transpose(p, (1, 2, 0))
        t = np.transpose(t, (1, 2, 0))
        score = sk_ssim(p, t, channel_axis=2, data_range=1.0)
        scores.append(score)
    return float(np.mean(scores))


def tensor_to_pil(t):
    """t: (3, H, W) tensor in [0, 1] -> PIL Image."""
    arr = (t.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def save_comparison_grid(lr_bicubic, pred, hr, path, max_rows=4):
    """Saves a PNG grid: rows = samples, columns = [input | prediction | target].
    lr_bicubic, pred, hr: (B, 3, H, W) tensors in [0, 1]."""
    n = min(lr_bicubic.shape[0], max_rows)
    thumb = hr.shape[-1]
    pad = 6
    cols = 3
    grid = Image.new("RGB", (thumb * cols + pad * (cols + 1),
                              (thumb + pad) * n + pad), color=(15, 15, 15))
    for row in range(n):
        imgs = [tensor_to_pil(lr_bicubic[row]), tensor_to_pil(pred[row]), tensor_to_pil(hr[row])]
        y = pad + row * (thumb + pad)
        for col, im in enumerate(imgs):
            x = pad + col * (thumb + pad)
            grid.paste(im, (x, y))
    grid.save(path)