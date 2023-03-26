import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class DepthwiseConv2d(ConvBlock):  # conv3x3
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, groups=out_channels, **kwargs)

class PointwiseConv2d(ConvBlock): # conv1x1 (= BottleNeck layer)
    def __init__(self, in_channels, out_channels, kernel_size=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, wise='depth_point', **kwargs):
        super().__init__()
        
        if wise == 'depth_point':
            self.wise1 = DepthwiseConv2d(in_channels, in_channels, kernel_size, padding, **kwargs)
            self.wise2 = PointwiseConv2d(in_channels, out_channels, act=kwargs.get('act', True))
        elif wise == 'point_depth':
            self.wise1 = PointwiseConv2d(in_channels, out_channels, act=kwargs.get('act', True))
            self.wise2 = DepthwiseConv2d(out_channels, out_channels, kernel_size, padding, **kwargs)
            
    def forward(self, x):
        return self.wise2(self.wise1(x))
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 28, 28)
    model = SeparableConv2d(3, 64, wise='point_depth')
    print(model(x).shape)