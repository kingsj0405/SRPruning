import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, [1,3], 1, [0,1], bias=True)
        self.conv2 = nn.Conv2d(nf, nf, [3,1], 1, [1,0], bias=True)
        # self.conv3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        # out = F.relu(self.conv2(out), inplace=True)
        out = self.conv2(out)
        return identity + out



class ESRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(ESRNet, self).__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, dilation=1)
        
        basic_block = functools.partial(ResidualBlock_noBN, nf=64)
        self.recon_trunk = make_layer(basic_block, 10)
        
        self.conv98 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv99 = nn.Conv2d(64, 32*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)


        self.HRconv = nn.Conv2d(32, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                m.weight.data *= 0.1  # for residual block
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        # B, C, H, W = x_in.size()
        # x_in = x_in.reshape(B*C, 1, H, W)

        x = self.conv1(x_in)
        x0 = x

        x = self.recon_trunk(x)
        x += x0
        
        x = self.conv98(x)  # B, 4*4*2, H, W
        x = self.conv99(F.relu(x, inplace=True))  # B, 4*4*2, H, W
        x = self.pixel_shuffle(x)   # B*C, 2, 4H, 4W

        x = self.HRconv(F.relu(x, inplace=True))  # B, 4*4*2, H, W
        x = self.conv_last(F.relu(x, inplace=True))  # B, 4*4*2, H, W

        return x