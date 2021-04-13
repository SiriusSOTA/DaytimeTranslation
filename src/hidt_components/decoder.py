from torch.nn import (
    Module,
    LeakyReLU,
    Tanh,
)

from .blocks import AdaResBlock, ConvBlock


class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.ada_res_block_1 = AdaResBlock(
            in_channels=128,
            out_channels=128,
        )
        self.act_1 = LeakyReLU()
        self.ada_res_block_2 = AdaResBlock(
            in_channels=128,
            out_channels=64,
        )
        self.act_2 = LeakyReLU()
        self.conv_block_3 = ConvBlock(
            in_channels=64,
            out_channels=32,
            transposed=True,
        )
        self.act_3 = LeakyReLU()
        self.ada_res_block_4 = AdaResBlock(
            in_channels=32,
            out_channels=16,
        )
        self.act_4 = LeakyReLU()
        self.ada_res_block_5 = AdaResBlock(
            in_channels=16,
            out_channels=16,
        )
        self.act_5 = LeakyReLU()
        self.conv_block_6 = ConvBlock(
            in_channels=16,
            out_channels=8,
            transposed=True,
        )
        self.act_6 = LeakyReLU()
        self.ada_res_block_7 = AdaResBlock(
            in_channels=8,
            out_channels=3,
        )
        self.act_7 = Tanh()

    def forward(self, content, style, hooks):
        x = self.ada_res_block_1(content, style, content)
        x = self.act_1(x)
        x = self.ada_res_block_2(x, style, hooks[3])
        x = self.act_2(x)
        x = self.conv_block_3(x)
        x = self.act_3(x)
        x = self.ada_res_block_4(x, style, hooks[2])
        x = self.act_4(x)
        x = self.ada_res_block_5(x, style, hooks[1])
        x = self.act_5(x)
        x = self.conv_block_6(x)
        x = self.act_6(x)
        x = self.ada_res_block_7(x, style, hooks[0])
        x = self.act_7(x)

        return x, None
