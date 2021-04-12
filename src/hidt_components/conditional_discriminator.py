import torch
from torch.nn import (
    Linear,
    Module,
    ReLU,
    Sequential,
)

from .blocks import ConvBlock, ResBlock


class ConditionalDiscriminator(Module):
    "Projection based discrminator, adapted from: https://github.com/XHChen0528/SNGAN_Projection_Pytorch"

    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat
        self.activation = ReLU()
        self.block_1 = ConvBlock(3, num_feat)
        self.blocks = Sequential(
            ResBlock(
                num_feat,
                num_feat,
                norm="layer",
            ),
            ConvBlock(
                num_feat,
                num_feat * (2 ** 1),
                stride=2,
            ),
            ResBlock(
                num_feat * (2 ** 1),
                num_feat * (2 ** 1),
                norm="layer",
            ),
            ConvBlock(
                num_feat * (2 ** 1),
                num_feat * (2 ** 2),
                stride=2,
            ),
            ResBlock(
                num_feat * (2 ** 2),
                num_feat * (2 ** 2),
                norm="layer",
            ),
            ConvBlock(
                num_feat * (2 ** 2),
                num_feat * (2 ** 3),
                stride=2,
            ),
            ResBlock(
                num_feat * (2 ** 3),
                num_feat * (2 ** 3),
                norm="layer",
            ),
            ConvBlock(
                num_feat * (2 ** 3),
                num_feat * (2 ** 4),
                stride=2,
            ),
        )

        self.l6 = torch.nn.utils.spectral_norm(
            Linear(num_feat * 16, 1)
        )
        self.style = torch.nn.utils.spectral_norm(
            Linear(3, num_feat * 16)
        )
        self._initialize()

    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            torch.nn.init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        x = self.block_1(x)
        x = self.blocks(x)

        h = self.activation(x)
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)

        if y is not None:
            output += torch.sum(self.style(y) * h, dim=1, keepdim=True)

        return output
