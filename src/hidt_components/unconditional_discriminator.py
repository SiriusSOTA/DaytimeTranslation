import torch
from torch.nn import (
    Flatten,
    Linear,
    Module,
    Sequential,
)

from .blocks import ConvBlock, ResBlock


class UnconditionalDiscriminator(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=64,
                stride=2,
            ),
            ResBlock(
                in_channels=64,
                out_channels=64,
                norm="layer",
            ),
            ConvBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
            ),
            ResBlock(
                in_channels=128,
                out_channels=128,
                norm="layer",
            ),
            ConvBlock(
                in_channels=128,
                out_channels=64,
                stride=2,
            ),
            ResBlock(
                in_channels=64,
                out_channels=64,
                norm="layer",
            ),
            ConvBlock(
                in_channels=64,
                out_channels=32,
                stride=2,
            ),
            ConvBlock(
                in_channels=32,
                out_channels=1,
            ),
            Flatten(),
            Linear(256, 1),
        )

    def forward(self, image):
        x = self.model(image)

        return x
