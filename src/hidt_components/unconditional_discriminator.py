from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    LeakyReLU,
    Tanh,
)
from torch.nn.utils import spectral_norm

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
            LeakyReLU(),
            ResBlock(
                in_channels=64,
                out_channels=64,
                norm="layer",
            ),
            LeakyReLU(),
            ConvBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
            ),
            LeakyReLU(),
            ResBlock(
                in_channels=128,
                out_channels=128,
                norm="layer",
            ),
            LeakyReLU(),
            ConvBlock(
                in_channels=128,
                out_channels=64,
                stride=2,
            ),
            LeakyReLU(),
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
            LeakyReLU(),
            spectral_norm(Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='refle ct',
            )),
        )

    def forward(self, image):
        x = self.model(image).squeeze(dim=1)

        return x
