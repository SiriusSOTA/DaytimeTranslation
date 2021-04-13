from torch.nn import Module, Sequential, LeakyReLU, Tanh

from .blocks import ConvBlock, ResBlock


class ContentEncoder(Module):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels=3,
                                      out_channels=8,
                                      norm="batch")
        self.act_1 = LeakyReLU()
        self.conv_block_2 = ConvBlock(in_channels=8,
                                      out_channels=16,
                                      norm="batch",
                                      stride=2)
        self.act_2 = LeakyReLU()
        self.conv_block_3 = ConvBlock(in_channels=16,
                                      out_channels=32,
                                      norm="batch")
        self.act_3 = LeakyReLU()
        self.conv_block_4 = ConvBlock(in_channels=32,
                                      out_channels=64,
                                      norm="batch",
                                      stride=2)
        self.act_4 = LeakyReLU()
        self.conv_block_5 = Sequential(ConvBlock(in_channels=64,
                                                 out_channels=128,
                                                 norm="batch"),
                                       ResBlock(in_channels=128,
                                                out_channels=128,
                                                norm="batch"))
        self.act_5 = Tanh()

    def forward(self, image):
        hooks = []

        x = self.act_1(self.conv_block_1(image))
        hooks.append(x)
        x = self.act_2(self.conv_block_2(x))
        hooks.append(x)
        x = self.act_3(self.conv_block_3(x))
        hooks.append(x)
        x = self.act_4(self.conv_block_4(x))
        hooks.append(x)
        x = self.act_5(self.conv_block_5(x))
        return x, hooks
