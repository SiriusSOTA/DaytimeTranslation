from torch.nn import Module, Sequential, LeakyReLU, Tanh

from .blocks import ConvBlock, ResBlock


class ContentEncoder(Module):
    def __init__(self):
        super().__init__()
        self.res_block_1 = Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=8,
            ),
            LeakyReLU(),
            ResBlock(
                in_channels=8,
                out_channels=8,
                norm="batch",
            ),
        )
        self.act_1 = LeakyReLU()
        self.conv_block_2 = ConvBlock(
            in_channels=8,
            out_channels=16,
            stride=2,
        )
        self.act_2 = LeakyReLU()
        self.res_block_3 = ResBlock(
            in_channels=16,
            out_channels=16,
            norm="batch",
        )
        self.act_3 = LeakyReLU()
        self.res_block_4 = Sequential(
            ConvBlock(
                in_channels=16,
                out_channels=32,
            ),
            LeakyReLU(),
            ResBlock(
                in_channels=32,
                out_channels=32,
                norm="batch",
            ),
        )
        self.act_4 = LeakyReLU()
        self.conv_block_5 = ConvBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
        )
        self.act_5 = LeakyReLU()
        self.res_block_6 = Sequential(
            ConvBlock(
                in_channels=64,
                out_channels=128,
            ),
            LeakyReLU(),
            ResBlock(
                in_channels=128,
                out_channels=128,
                norm="batch",
            ),
        )
        self.act_6 = LeakyReLU()
        self.conv_block_7 = ConvBlock(
            in_channels=128,
            out_channels=128,
        )
        self.act_7 = Tanh()

    def forward(self, image):
        hooks = []

        x = self.act_1(self.res_block_1(image))
        hooks.append(x)
        x = self.act_2(self.conv_block_2(x))
        x = self.act_3(self.res_block_3(x))
        hooks.append(x)
        x = self.act_4(self.res_block_4(x))
        hooks.append(x)
        x = self.act_5(self.conv_block_5(x))
        x = self.act_6(self.res_block_6(x))
        hooks.append(x)
        x = self.act_7(self.conv_block_7(x))
        return x, hooks
