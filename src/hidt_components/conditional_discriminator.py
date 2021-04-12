import torch
from torch.nn import (
    Linear,
    Module,
    LeakyReLU,
    Sequential,
    Conv2d,
)

from .blocks import ConvBlock, ResBlock


class ConditionalDiscriminator(Module):
    "Projection based discrminator, adapted from: https://github.com/XHChen0528/SNGAN_Projection_Pytorch"

    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat
        self.activation = LeakyReLU()
        self.block_1 = ConvBlock(6, num_feat)
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

        self.block_3 = torch.nn.utils.spectral_norm(
            Conv2d(in_channels=num_feat * 16,
                   out_channels=1,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   padding_mode='reflect')
        )

    def forward(self, x, y):
        y = y.repeat_interleave(x.shape[2] * x.shape[3]).view([-1, 3, x.shape[2], x.shape[3]])
        x = torch.cat([x, y], axis=1)
        x = self.block_1(x)
        x = self.blocks(x)

        h = self.activation(x)
        output = self.block_3(h).squeeze(dim=1)

        return output
