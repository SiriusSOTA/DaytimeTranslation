import torch
from einops.layers.torch import Rearrange
from torch.nn import (
    AdaptiveAvgPool2d,
    Module,
    Sequential,
)

from .blocks import ConvBlock


class StyleEncoder(Module):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=8,
                stride=2,
                norm="batch",
            ),
            ConvBlock(
                in_channels=8,
                out_channels=16,
                stride=2,
                norm="batch",
            ),
            ConvBlock(
                in_channels=16,
                out_channels=32,
                stride=2,
                norm="batch",
            ),
            ConvBlock(
                in_channels=32,
                out_channels=3,
                stride=2,
                norm="batch",
            ),
            AdaptiveAvgPool2d(output_size=1),
            Rearrange('b c 1 1 -> b c'),
        )

    def forward(self, image):
        x = self.encoder(image)
        return x
