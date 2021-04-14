from einops.layers.torch import Rearrange
from torch.nn import (
    AdaptiveAvgPool2d,
    Module,
    Sequential,
    LeakyReLU,
    Conv2d,
    ModuleList,
)

from .blocks import Conv2dBlock, NormalizeOutput


class StyleEncoderBase(Module):
    def __init__(self, dim):
        super().__init__()
        self.output_dim = dim
        self.body = ModuleList()
        self.head = ModuleList()

    def forward(self, tensor, spade_input=None):
        if spade_input:
            for layer in self.body:
                tensor = layer(tensor, spade_input)
        else:
            for layer in self.body:
                tensor = layer(tensor)

        for layer in self.head:
            tensor = layer(tensor)

        return tensor


class StyleEncoder(StyleEncoderBase):
    def __init__(self, num_downsamples, input_dim, dim, output_dim, norm, activ, pad_type, normalized_out=False):
        super().__init__(dim)
        self.body += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.body += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(num_downsamples - 2):
            self.body += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.head += [AdaptiveAvgPool2d(1)]
        self.head += [Conv2d(dim, output_dim, 1, 1, 0)]
        if normalized_out:
            self.head += [NormalizeOutput(dim=1)]