import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels= 3,
            out_channels= 64,
            kernel_size= 5,
            padding= 2
        )

        self.conv2 = nn.Conv2d(
            in_channels= 64,
            out_channels= 32,
            kernel_size= 3,
            padding= 1
        )

        self.conv3 = nn.Conv2d(
            in_channels= 32,
            out_channels= 48,
            kernel_size= 3,
            padding= 1
        )

        self.relu = nn.ReLU(inplace= True)

        self.pixel_shuffle = nn.PixelShuffle(
            upscale_factor= 4
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x
