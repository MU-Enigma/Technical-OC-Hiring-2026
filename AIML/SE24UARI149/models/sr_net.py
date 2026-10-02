"""
sr_net.py

A small, configurable super-resolution network designed around CPU/limited-GPU
compute constraints, following the design reasoning laid out in the project
write-up:

  1. Almost all convolutions operate in LOW-resolution space (on the input
     LR image directly), not on an upsampled version of it. This is the
     single biggest compute lever: cost scales with spatial size, and doing
     the heavy feature extraction at LR resolution avoids paying a (scale
     factor)^2 penalty on every conv layer, unlike SRCNN/VDSR-style networks
     that upsample first and then convolve.

  2. Upsampling to HR resolution happens only at the very end via
     PixelShuffle (sub-pixel convolution): the network learns C*r^2 channels
     at LR resolution and a parameter-free reshuffle op rearranges them into
     r times more spatial resolution. This is far cheaper than learned
     deconvolution/transposed-conv upsampling and avoids checkerboard
     artifacts associated with strided transposed convolutions.

  3. Residual blocks (conv -> ReLU -> conv -> skip-add), no BatchNorm.
     BatchNorm is deliberately omitted: it normalizes per-batch statistics,
     which discards the absolute brightness/intensity information the
     network needs for faithful reconstruction -- a well-documented finding
     from EDSR. Removing it also reduces compute.

  4. Global residual-on-bicubic skip: the network predicts a residual
     correction on top of a plain bicubic upsample of the input, rather than
     reconstructing the whole image from scratch. This lets the network
     spend its capacity on high-frequency detail instead of relearning
     trivial low-frequency content that simple interpolation already gets
     right, and empirically speeds up convergence (VDSR's core idea).

Configurable via JSON (see configs/model_tiny.json, configs/model_small.json):
    - num_blocks: number of residual blocks in the body
    - num_channels: channel width throughout the body
    - scale_factor: must match the degradation pipeline's scale_factor
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block: conv -> ReLU -> conv -> skip-add.
    No BatchNorm (see module docstring for justification)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity


class PixelShuffleUpsampler(nn.Module):
    """Upsamples by an integer scale_factor using sub-pixel convolution.

    For scale_factor not in {2, 3, 4, 8}, this decomposes into stages of
    factor-2 upsampling where possible, which is the standard approach used
    by ESPCN/EDSR-style networks since PixelShuffle natively expects a
    single integer squared-channel expansion.
    """

    def __init__(self, channels: int, scale_factor: int):
        super().__init__()
        layers = []
        remaining = scale_factor
        # Decompose into factor-2 steps if possible (covers 2, 4, 8...).
        while remaining % 2 == 0:
            layers.append(nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
            remaining //= 2
        if remaining != 1:
            # Handles factor-3 (or other leftover) directly.
            layers.append(nn.Conv2d(channels, channels * (remaining ** 2),
                                     kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(remaining))
            layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class SRNet(nn.Module):
    def __init__(self, num_blocks: int = 8, num_channels: int = 64,
                 scale_factor: int = 4, in_channels: int = 3):
        super().__init__()
        self.scale_factor = scale_factor

        # Stem: lift raw LR image into the working channel width.
        self.stem = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.stem_relu = nn.ReLU(inplace=True)

        # Body: stack of residual blocks operating entirely at LR resolution.
        self.body = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])

        # A conv after the residual stack, with its own skip from the stem
        # output (a common EDSR-style pattern that stabilizes deep residual
        # stacks by giving the body an explicit "correction" role).
        self.body_tail_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Upsampling head: only place spatial resolution actually increases.
        self.upsampler = PixelShuffleUpsampler(num_channels, scale_factor)

        # Final conv maps back to 3 (RGB) channels, producing the residual
        # detail to add on top of the bicubic-upsampled input.
        self.out_conv = nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, lr):
        # Global residual base: plain bicubic upsample of the raw input.
        # The network only needs to learn the *correction* on top of this,
        # not the whole image from scratch (see module docstring, point 4).
        bicubic_base = F.interpolate(
            lr, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )

        feat = self.stem_relu(self.stem(lr))
        body_out = self.body(feat)
        body_out = self.body_tail_conv(body_out)
        feat = feat + body_out  # stem-to-tail skip around the whole residual stack

        upsampled_feat = self.upsampler(feat)
        residual_detail = self.out_conv(upsampled_feat)

        out = bicubic_base + residual_detail
        return torch.clamp(out, 0.0, 1.0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(config_path: str) -> SRNet:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = SRNet(
        num_blocks=cfg["num_blocks"],
        num_channels=cfg["num_channels"],
        scale_factor=cfg["scale_factor"],
    )
    return model, cfg


if __name__ == "__main__":
    import sys
    import time

    config_path = sys.argv[1] if len(sys.argv) > 1 else "../configs/model_tiny.json"
    model, cfg = build_model_from_config(config_path)
    print(f"Config: {cfg}")
    print(f"Parameters: {model.count_parameters():,}")

    batch = torch.randn(8, 3, 32, 32).clamp(0, 1)
    t0 = time.time()
    out = model(batch)
    fwd_time = time.time() - t0
    print(f"Input: {tuple(batch.shape)} -> Output: {tuple(out.shape)}, "
          f"forward pass took {fwd_time:.4f}s for batch of 8")

    # Quick backward pass timing too, since that's ~2x forward cost typically.
    target = torch.randn_like(out).clamp(0, 1)
    loss_fn = nn.L1Loss()
    t0 = time.time()
    loss = loss_fn(out, target)
    loss.backward()
    bwd_time = time.time() - t0
    print(f"Backward pass took {bwd_time:.4f}s, loss={loss.item():.4f}")
