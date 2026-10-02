import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicBaseline(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, lr_bicubic):
        return lr_bicubic


class SRCNN(nn.Module):
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
        return self.net(lr_bicubic)


class ResidualBlock(nn.Module):
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
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.shuffle(self.conv(x)))


class SRResNetLite(nn.Module):
    def __init__(self, scale=4, channels=64, num_blocks=16):
        super().__init__()
        assert scale in (2, 4, 8)
        self.scale = scale
        n_upsample_stages = int(math.log2(scale))
        self.entry = nn.Sequential(nn.Conv2d(3, channels, kernel_size=9, padding=4), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.post_block_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(*[PixelShuffleStage(channels) for _ in range(n_upsample_stages)])
        self.exit = nn.Conv2d(channels, 3, kernel_size=9, padding=4)

    def forward(self, lr_small, lr_bicubic):
        feat = self.entry(lr_small)
        out = self.blocks(feat)
        out = self.post_block_conv(out) + feat
        out = self.upsample(out)
        delta = self.exit(out)
        if delta.shape[-2:] != lr_bicubic.shape[-2:]:
            delta = F.interpolate(delta, size=lr_bicubic.shape[-2:], mode="bilinear", align_corners=False)
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
