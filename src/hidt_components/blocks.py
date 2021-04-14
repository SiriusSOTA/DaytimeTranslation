from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Identity,
    InstanceNorm2d,
    Linear,
    MaxPool2d,
    Module,
    Sequential,
    LayerNorm,
    LeakyReLU,
    Parameter,
)
import torch.nn.functional as F

from config.hidt_config import config


class BilinearInterpolation(nn.Module):
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True, recompute_scale_factor=True)
        return x


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
            padding_mode: str = "reflect",
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
            padding_mode: str = "reflect"
    ):
        block_ordered_dict = OrderedDict()
        block_ordered_dict["conv"] = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        ) if not transposed else Sequential(
            BilinearInterpolation(),
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
            )
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
        self.conv_block = Sequential(block_ordered_dict).to(config["device"])

    def forward(self, x):
        if self.conv_block is None:
            self._init_block(x.size(), **self.parameters)
        x = self.conv_block(x)
        return x

    
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', style_dim=3, norm_after_conv='ln'):
        super().__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        self.compute_kernel = True if norm == 'conv_kernel' else False
        self.WCT = True if norm == 'WCT' else False

        norm_dim = output_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'WCT':
            self.norm = nn.InstanceNorm2d(norm_dim)
            self.style_dim = style_dim
            self.dim = output_dim, input_dim, kernel_size, kernel_size
            self.output_dim = output_dim
            self.stride = stride
            self.mlp_W = nn.Sequential(
                nn.Linear(self.style_dim, output_dim**2),
            )
            self.mlp_bias = nn.Sequential(
                nn.Linear(self.style_dim, output_dim),
            )
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'conv_kernel':
            self.style_dim = style_dim
            self.norm_after_conv = norm_after_conv
            self._get_norm(self.norm_after_conv, norm_dim)
            self.dim = output_dim, input_dim, kernel_size, kernel_size
            self.stride = stride
            self.mlp_kernel = nn.Linear(self.style_dim, int(np.prod(self.dim)))
            self.mlp_bias = nn.Linear(self.style_dim, output_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.style = None

    def _get_norm(self, norm, norm_dim):
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

    def forward(self, x, spade_input=None):
        if self.compute_kernel:
            conv_kernel = self.mlp_kernel(self.style)
            conv_bias = self.mlp_bias(self.style)
            x = F.conv2d(self.pad(x), conv_kernel.view(*self.dim), conv_bias.view(-1), self.stride)
        else:
            x = self.conv(self.pad(x))
        if self.WCT:
            x_mean = x.mean(-1).mean(-1)
            x = x.permute(0, 2, 3, 1)
            x = x - x_mean
            W = self.mlp_W(self.style)
            bias = self.mlp_bias(self.style)
            W = W.view(self.output_dim, self.output_dim)
            x = x @ W
            x = x + bias
            x = x.permute(0, 3, 1, 2)
        if self.norm:
            if self.norm_type == 'spade':
                x = self.norm(x, spade_input)
            else:
                x = self.norm(x)
        if self.activation:
            x = self.activation(x)

        return x


class ResBlock(Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', style_dim=3, norm_after_conv='ln',
                 res_off=False):
        super().__init__()
        self.res_off = res_off
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type,
                              style_dim=style_dim, norm_after_conv=norm_after_conv)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type,
                              style_dim=style_dim, norm_after_conv=norm_after_conv)]
        self.model = nn.ModuleList(model)

    def forward(self, x, spade_input=None):
        residual = x
        for layer in self.model:
            x = layer(x, spade_input)
        if self.res_off:
            return x
        else:
            return x + residual

    
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', non_local=False,
                 style_dim=3, norm_after_conv='ln'):
        super(ResBlocks, self).__init__()
        self.model = []
        if isinstance(non_local, (list,)):
            for i in range(num_blocks):
                if i in non_local:
                    raise DeprecationWarning()
                else:
                    self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type,
                                            style_dim=style_dim, norm_after_conv=norm_after_conv)]
        else:
            for i in range(num_blocks):
                self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type,
                                        style_dim=style_dim, norm_after_conv=norm_after_conv)]

        self.model = Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    
    
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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.ada_creator = nn.Sequential(
            Linear(in_features=3, out_features=16),
            Linear(in_features=16, out_features=64),
            Linear(in_features=64, out_features=256),
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
        self.act_1 = LeakyReLU() # TODO find out why cant be inplace=True
        self.res_block = Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm="batch",
            ),
            LeakyReLU(), # TODO find out why cant be inplace=True
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
                norm="batch",
            )
        else:
            self.skip = Identity()

    def forward(self, content, style, hook):
        ada = self.act_1(self.ada_block(content, style, hook))
        res = self.res_block(ada)

        if self.skip is not None:
            content = self.skip(content)

        return res + content
    
    
class NormalizeOutput(Module):
    """
    Module that scales the input tensor to the unit norm w.r.t. the specified axis.
    Actually, the module analog of `torch.nn.functional.normalize`
    """

    def __init__(self, dim=1, p=2, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.p = p

    def forward(self, tensor):
        return F.normalize(tensor, p=self.p, dim=self.dim, eps=self.eps)

    
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
    
    
class AdaptiveInstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x