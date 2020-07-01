import torch
import torch.nn as nn
import model.CARN.ops as ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(in_channels, out_channels)
        self.b2 = ops.ResidualBlock(in_channels, out_channels)
        self.b3 = ops.ResidualBlock(in_channels, out_channels)
        self.c1 = ops.BasicBlock(in_channels*2, out_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(in_channels*3, out_channels, 1, 1, 0)
        self.c3 = ops.BasicBlock(in_channels*4, out_channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        channel_cnt = kwargs.get("channel_cnt", 64)
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, channel_cnt, 3, 1, 1)

        self.b1 = Block(channel_cnt, channel_cnt)
        self.b2 = Block(channel_cnt, channel_cnt)
        self.b3 = Block(channel_cnt, channel_cnt)
        self.c1 = ops.BasicBlock(channel_cnt*2, channel_cnt, 1, 1, 0)
        self.c2 = ops.BasicBlock(channel_cnt*3, channel_cnt, 1, 1, 0)
        self.c3 = ops.BasicBlock(channel_cnt*4, channel_cnt, 1, 1, 0)
        
        self.upsample = ops.UpsampleBlock(channel_cnt, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(channel_cnt, 3, 3, 1, 1)
                
    def forward(self, x, scale=4):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out