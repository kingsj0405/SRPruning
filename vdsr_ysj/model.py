import torch.nn as nn


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        # Make layers
        self.input = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.relu = nn.ReLU()
        self.residual_layer = self._make_layer(Block, 18)
        self.output = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        # Initialize weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.ModuleList(layers)

    def forward(self, x):
        out = self.relu(self.input(x))
        for l in self.residual_layer:
            out = l(out)
        out = self.output(out)
        return out + x
