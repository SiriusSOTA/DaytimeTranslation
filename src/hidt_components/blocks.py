from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Identity,
    InstanceNorm2d,
    Linear,
    MaxPool2d,
    Module,
    LeakyReLU,
    Sequential,
    LayerNorm,
)

from config.hidt_config import config


class ConvBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = True,
            transposed: bool = False,
            norm: Optional[str] = None,
            pool: bool = False,
            act: bool = True,
            padding_mode: str = "zeros"
    ):
        super().__init__()
        self.conv_block = None
        self.parameters = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias,
            "transposed": transposed,
            "norm": norm,
            "pool": pool,
            "act": act,
            "padding_mode": padding_mode,
        }

    def _init_block(
            self,
            input_shape,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = True,
            transposed: bool = False,
            norm: Optional[str] = None,
            pool: bool = False,
            act: bool = True,
            padding_mode: str = "zeros"
    ):
        block_ordered_dict = OrderedDict()
        block_ordered_dict['conv'] = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        ) if not transposed else ConvTranspose2d(  # TODO: bilinear?
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        if norm is not None:
            if norm == "batch":
                block_ordered_dict['norm'] = BatchNorm2d(
                    num_features=out_channels)
            elif norm == "layer":
                block_ordered_dict['norm'] = LayerNorm(input_shape[1:])
            elif norm == "instance":
                block_ordered_dict['norm'] = InstanceNorm2d(
                    num_features=out_channels,
                    affine=True,
                    track_running_stats=True,
                )

        if pool:
            block_ordered_dict['pool'] = MaxPool2d(kernel_size=2)
        if act:
            block_ordered_dict['act'] = LeakyReLU()
        self.conv_block = Sequential(block_ordered_dict).to(config["device"])

    def forward(self, x):
        if self.conv_block is None:
            self._init_block(x.size(), **self.parameters)
        x = self.conv_block(x)
        return x


class ResBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            norm: Optional[str] = None,
            padding: int = 1,
            padding_mode: str = "zeros"
    ):
        super().__init__()
        self.res_block = Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                norm=norm,
                padding_mode=padding_mode,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                norm=norm,
                act=False,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


def get_adastats(features):
    bs, c = features.shape[:2]
    features = features.view(bs, c, -1)
    mean = features.mean(dim=2).view(bs, c, 1, 1)
    std = features.var(dim=2).sqrt().view(bs, c, 1, 1)
    return mean, std


def AdaIN(content_feat, style_feat):
    # calculating channel and batch specific stats
    smean, sstd = get_adastats(style_feat)
    cmean, cstd = get_adastats(content_feat)
    csize = content_feat.size()
    norm_content = (
            (content_feat - cmean.expand(csize)) /
            (cstd.expand(csize) + 1e-3)
    )
    return norm_content * sstd.expand(csize) + smean.expand(csize)


class AdaSkipBlock(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ada_creator = nn.Sequential(
            Linear(
                in_features=3,
                out_features=16,
            ),
            Linear(
                in_features=16,
                out_features=64,
            ),
            Linear(
                in_features=64,
                out_features=256,
            ),
        )
        self.ada = AdaIN
        self.dense = ConvBlock(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            norm="batch",
        )

    def forward(self, content, style, hook):
        x = self.ada_creator(style)
        ada_params = x.view((x.shape[0], self.in_channels, -1))
        ada = self.ada(hook, ada_params)
        combined = torch.cat([content, ada], dim=1)
        x = self.dense(combined)
        return x


class AdaResBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.ada_block = AdaSkipBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.res_block = Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm="batch",
            ),
            ResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                norm="batch",
            ),
        )
        if in_channels != out_channels:
            self.skip = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                act=False,
                norm="batch",
            )
        else:
            self.skip = Identity()

    def forward(self, content, style, hook):
        ada = self.ada_block(content, style, hook)
        res = self.res_block(ada)

        if self.skip is not None:
            content = self.skip(content)

        return res + content
