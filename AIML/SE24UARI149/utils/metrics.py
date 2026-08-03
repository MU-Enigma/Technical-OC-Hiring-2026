"""
metrics.py

Standard super-resolution quality metrics: PSNR and SSIM.

Both operate on tensors already in [0, 1] float range, shape (B, C, H, W) or
(C, H, W). These are the field-standard quantitative metrics for SR, but they
are known to correlate imperfectly with human-perceived quality -- a blurry,
"safe" output can score well on both while looking visibly worse than a
sharper output with slightly different local pixel values. This is why
evaluate.py also produces qualitative image grids and a bicubic-baseline
comparison rather than relying on these numbers alone (see project write-up,
"Evaluation and Analysis").
"""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio, averaged over the batch.
    Higher is better. Measures pixel-wise fidelity; sensitive to blur less
    than to any shift/misalignment or color error.
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)  # per-sample MSE
    mse = torch.clamp(mse, min=1e-10)  # avoid log(0) for a perfect match
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.outer(g)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         max_val: float = 1.0) -> torch.Tensor:
    """Structural Similarity Index, averaged over the batch.
    Higher is better (max 1.0). Captures luminance/contrast/structure
    similarity rather than pure pixel error, closer to (but still not a
    perfect proxy for) perceived quality.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma=1.5, channels=channels,
                               device=pred.device, dtype=pred.dtype)

    pad = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_target

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = (
        ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) /
        ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    )
    return ssim_map.flatten(1).mean(dim=1)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Returns mean PSNR and SSIM over the batch as plain Python floats."""
    with torch.no_grad():
        pred_c = torch.clamp(pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        psnr_val = psnr(pred_c, target_c).mean().item()
        ssim_val = ssim(pred_c, target_c).mean().item()
    return {"psnr": psnr_val, "ssim": ssim_val}


if __name__ == "__main__":
    # Smoke test: identical images should give PSNR=inf-ish (very high) and SSIM=1.0
    a = torch.rand(4, 3, 64, 64)
    b = a.clone()
    print("Identical images:", compute_metrics(a, b))

    b_noisy = torch.clamp(a + torch.randn_like(a) * 0.05, 0, 1)
    print("Slightly noisy:", compute_metrics(a, b_noisy))

    b_random = torch.rand(4, 3, 64, 64)
    print("Unrelated random images:", compute_metrics(a, b_random))
