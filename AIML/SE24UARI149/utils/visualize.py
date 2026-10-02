"""
visualize.py

Saves side-by-side (bicubic-upsampled input | model output | ground-truth
target) image grids at regular training intervals. This is what satisfies
the "training progression images" deliverable -- a visual record of how the
model's output quality evolves over the course of training, not just a loss
curve.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid


def save_progression_grid(lr_batch: torch.Tensor, pred_batch: torch.Tensor,
                           hr_batch: torch.Tensor, out_path: str,
                           scale_factor: int, max_samples: int = 4):
    """lr_batch, pred_batch, hr_batch: (B, C, H, W) tensors in [0, 1].
    Produces a grid with rows = [bicubic-upsampled input, model output, ground truth]
    for up to `max_samples` examples from the batch.
    """
    n = min(max_samples, lr_batch.shape[0])
    lr = lr_batch[:n]
    pred = torch.clamp(pred_batch[:n], 0.0, 1.0)
    hr = hr_batch[:n]

    bicubic_up = F.interpolate(lr, scale_factor=scale_factor, mode="bicubic",
                                align_corners=False)
    bicubic_up = torch.clamp(bicubic_up, 0.0, 1.0)

    # Interleave as (sample0: bicubic, pred, hr), (sample1: bicubic, pred, hr), ...
    rows = []
    for i in range(n):
        rows.append(bicubic_up[i])
        rows.append(pred[i])
        rows.append(hr[i])
    grid_tensor = torch.stack(rows, dim=0)

    grid = make_grid(grid_tensor, nrow=3, padding=4, pad_value=1.0)
    grid_img = Image.fromarray(
        (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_path)
    return out_path
