"""
discriminator.py

A small PatchGAN-style discriminator for adversarial SR training.

Rather than a single real/fake judgment for the whole image, this outputs a
grid of real/fake predictions, each corresponding to a local patch of the
input -- this is the standard PatchGAN design (Isola et al., pix2pix, and
used in SRGAN's discriminator too): it's cheaper than a full-image
classifier, and it pushes the generator toward getting LOCAL texture/detail
right everywhere, which is exactly what's missing from a pure pixel-loss
model (see WRITEUP.md's discriminative-vs-generative discussion).

Deliberately kept small relative to the generator (SRNet) -- an
overly powerful discriminator learns to distinguish real/fake trivially
and stops giving the generator a useful learning signal, a common and
well-documented GAN failure mode.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, stride, use_norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
    if use_norm:
        # InstanceNorm, not BatchNorm: discriminator operates on a mix of
        # real HR patches and generator outputs, and BatchNorm's per-batch
        # statistics can behave inconsistently across that mix; InstanceNorm
        # normalizes each sample independently, avoiding that issue.
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            conv_block(in_channels, c, stride=1, use_norm=False),  # no norm on first layer (standard practice)
            conv_block(c, c, stride=2),
            conv_block(c, c * 2, stride=1),
            conv_block(c * 2, c * 2, stride=2),
            conv_block(c * 2, c * 4, stride=1),
            conv_block(c * 4, c * 4, stride=2),
            nn.Conv2d(c * 4, 1, kernel_size=3, stride=1, padding=1),
            # No final activation/sigmoid here -- LSGAN loss (see losses.py)
            # operates on raw logits directly for training stability.
        )

    def forward(self, x):
        return self.net(x)  # shape: (B, 1, H/8, W/8) -- a grid of real/fake logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PatchDiscriminator()
    print(f"Discriminator parameters: {model.count_parameters():,}")
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    print(f"Input: {tuple(x.shape)} -> Output (patch logits): {tuple(out.shape)}")