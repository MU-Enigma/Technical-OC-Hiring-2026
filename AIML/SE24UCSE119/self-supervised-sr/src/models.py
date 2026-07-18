"""
models.py
---------
Three models, forming the comparison required by the task:

1. Bicubic  - not a neural net. Pure interpolation. The floor every
              learned model must beat.
2. SRCNN    - Dong et al., 2014/2015. Faithful reconstruction: 9x9 patch
              extraction -> 1x1 non-linear mapping -> 5x5 reconstruction,
              predicting raw pixels directly (no residual connection -
              that idea belongs to VDSR, used only in SRResNet-lite below,
              so each borrowed idea stays attributed to one model).
              Pre-upsampling: operates on the input already bicubic-
              upsampled to full size.
3. SRResNetLite - Ledig et al. 2017 (SRResNet, the generator half of
              SRGAN), with:
                - PixelShuffle upsampling (Shi et al., ESPCN), done in
                  staged 2x steps (log2(scale) stages) rather than one
                  jump to the full scale factor, matching the actual
                  SRResNet architecture description.
                - no batch norm in residual blocks (Lim et al., EDSR
                  finding: BN normalizes away the pixel-intensity info SR
                  needs).
                - residual/delta prediction on top of a bicubic upsample
                  (Kim et al., VDSR: predicting HR-bicubic delta instead
                  of raw pixels is what lets deeper networks train well).
              Post-upsampling: operates on the small LR image internally,
              upsamples only at the end.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicBaseline(nn.Module):
    """Not learned. Included so 'is the model even better than free
    resizing' is answerable directly instead of assumed."""

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, lr_bicubic):
        return lr_bicubic


class SRCNN(nn.Module):
    """
    Dong et al., 'Image Super-Resolution Using Deep Convolutional Networks'.
    Three conv layers, matching the paper's actual kernel sizes:
      9x9 (patch extraction) -> 1x1 (non-linear mapping) -> 5x5 (reconstruction)
    Operates on the pre-upsampled (already full-size) input, predicts raw
    pixels directly - no residual connection, matching the original paper.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 3, kernel_size=5, padding=2),
        )

    def forward(self, lr_bicubic):
        return self.net(lr_bicubic)  # direct pixel prediction, no residual


class ResidualBlock(nn.Module):
    """Two conv layers + ReLU, with a skip connection. No BatchNorm -
    Lim et al. (EDSR) found BN hurts SR quality because it normalizes away
    the exact pixel-intensity information the task needs to preserve."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class PixelShuffleStage(nn.Module):
    """
    One sub-pixel convolution stage (Shi et al., ESPCN), upscaling by a
    factor of 2. Convolves up to 4x the channels, then PixelShuffle
    rearranges those extra channels into spatial resolution - no kernel
    overlap, no checkerboard artifacts (unlike transposed convolution).
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.shuffle(self.conv(x)))


class SRResNetLite(nn.Module):
    """
    Final model. Post-upsampling architecture:
      small LR (3, H/scale, W/scale)
        -> conv feature extraction
        -> N residual blocks (no BN)
        -> log2(scale) staged 2x PixelShuffle upsamples, matching Ledig
           et al.'s sequential 2x upsampling rather than one jump to scale
        -> conv to 3 channels  =>  predicted delta
      output = bicubic_upsample(LR) + predicted delta

    'Lite' relative to the original SRResNet paper (16 residual blocks,
    larger images): fewer blocks/channels, matched to Colab-T4-scale
    compute, while keeping every architectural idea that matters.
    """

    def __init__(self, scale=4, channels=32, num_blocks=6):
        super().__init__()
        assert scale in (2, 4, 8), \
            "designed for power-of-2 scales, matching Ledig et al.'s staged 2x upsampling"
        self.scale = scale
        n_upsample_stages = int(math.log2(scale))

        self.entry = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.post_block_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            *[PixelShuffleStage(channels) for _ in range(n_upsample_stages)]
        )
        self.exit = nn.Conv2d(channels, 3, kernel_size=9, padding=4)

    def forward(self, lr_small, lr_bicubic):
        feat = self.entry(lr_small)
        out = self.blocks(feat)
        out = self.post_block_conv(out) + feat  # global residual within the trunk
        out = self.upsample(out)
        delta = self.exit(out)

        # Defensive resize in case crop size isn't perfectly divisible by scale
        if delta.shape[-2:] != lr_bicubic.shape[-2:]:
            delta = F.interpolate(delta, size=lr_bicubic.shape[-2:],
                                   mode="bilinear", align_corners=False)

        return lr_bicubic + delta


def build_model(name, scale=4):
    name = name.lower()
    if name == "bicubic":
        return BicubicBaseline(scale=scale)
    elif name == "srcnn":
        return SRCNN()
    elif name in ("srresnet", "srresnet_lite", "srresnetlite"):
        return SRResNetLite(scale=scale)
    else:
        raise ValueError(f"Unknown model name: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)