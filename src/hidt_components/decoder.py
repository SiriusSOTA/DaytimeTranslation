from torch.nn import Module

from .blocks import AdaResBlock, ConvBlock


class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.ada_res_block_1 = AdaResBlock(
            in_channels=128,
            out_channels=128,
        )
        self.ada_res_block_2 = AdaResBlock(
            in_channels=128,
            out_channels=64,
        )
        self.conv_block_3 = ConvBlock(
            in_channels=64,
            out_channels=32,
            transposed=True,
        )
        self.ada_res_block_4 = AdaResBlock(
            in_channels=32,
            out_channels=16,
        )
        self.ada_res_block_5 = AdaResBlock(
            in_channels=16,
            out_channels=16,
        )
        self.conv_block_6 = ConvBlock(
            in_channels=16,
            out_channels=8,
            transposed=True,
        )
        self.ada_res_block_7 = AdaResBlock(
            in_channels=8,
            out_channels=3,
        )

    def forward(self, content, style, hooks):
        print("content", content.shape, "style", style.shape)
        x = self.ada_res_block_1(content, style, content)
        x = self.ada_res_block_2(x, style, hooks[3])
        print("before conv", x.shape)
        x = self.conv_block_3(x)
        print("after conv", x.shape)
        x = self.ada_res_block_4(x, style, hooks[2])
        x = self.ada_res_block_5(x, style, hooks[1])
        x = self.conv_block_6(x)
        print(x.shape)
        x = self.ada_res_block_7(x, style, hooks[0])
        print(x.shape)

        return x, None
