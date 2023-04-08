'https://arxiv.org/pdf/1807.11626.pdf'

import math
import torch
import torch.nn as nn
from point_depth_separable_conv import ConvBlock, SeparableConv2d

mnasnet_architecture = [
    # (out_channels, kernel_size, stride, padding, expansion, num_repeats, SE_block)
    (24, 3, 2, 1, 6, 2, False),
    (40, 5, 2, 2, 3, 3, True),
    (80, 3, 2, 1, 6, 4, False),
    (112, 3, 1, 1, 6, 2, True),
    (160, 5, 2, 2, 6, 3, True),
    (320, 3, 1, 1, 6, 1, False),
]

class PointwiseConv2d(ConvBlock): # conv1x1 (= BottleNeck layer)
    def __init__(self, in_channels, out_channels, kernel_size=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

class DepthwiseConv2d(ConvBlock):  # conv3x3
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, groups=out_channels, **kwargs)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=4) -> None:
        super().__init__()

        reduction_channels = math.ceil(channels / reduction_ratio)
        self.squeeze_excitation_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, reduction_channels, kernel_size=1),  # same as nn.Linear(gate_channels, channels)
            nn.ReLU(),
            nn.Conv2d(reduction_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.squeeze_excitation_layers(x)
    
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expansion=6, SE_block=True) -> None:
        super().__init__()

        self.residual = (in_channels == out_channels)
        hidden_channels = expansion * in_channels
        act = (stride == 1)

        if SE_block:
            self.bottleneck_layers = nn.Sequential(
                PointwiseConv2d(in_channels, hidden_channels),
                DepthwiseConv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding),
                SEBlock(hidden_channels),
                PointwiseConv2d(hidden_channels, out_channels, act=act),
            )
        else:
            self.bottleneck_layers = nn.Sequential(
                PointwiseConv2d(in_channels, hidden_channels),
                DepthwiseConv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding),
                PointwiseConv2d(hidden_channels, out_channels, act=act),
            )

    def forward(self, x):
        if self.residual:
            return x + self.bottleneck_layers(x)
        else:
            return self.bottleneck_layers(x)
        
class MnasNet(nn.Module):
    def __init__(self, config=mnasnet_architecture, in_channels=3, num_classes=1000) -> None:
        super().__init__()

        self.first_conv_layer = ConvBlock(in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.sep_conv_layer = SeparableConv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.mb_conv_layers, channels = self._create_layers(config)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_conv_layer = nn.Conv2d(in_channels=channels, out_channels=num_classes, kernel_size=1)

    def _create_layers(self, architecture):
        layers = []
        in_channels = 16

        for x in architecture:
            out_channels, kernel_size, stride, padding, expansion, num_repeats, SE_block = x

            layers += [MBConvBlock(in_channels, out_channels, kernel_size, stride, padding, expansion, SE_block)]
            layers += [MBConvBlock(out_channels, out_channels, kernel_size, 1, padding, expansion, SE_block) for _ in range(num_repeats-1)]
            in_channels = out_channels

        return nn.Sequential(*layers), out_channels
    
    def forward(self, x):
        x = self.first_conv_layer(x)
        x = self.sep_conv_layer(x)
        x = self.mb_conv_layers(x)
        
        x = self.avg_pool(x)
        x = self.last_conv_layer(x)
        x = x.flatten(start_dim=1)

        return x
    
if __name__=='__main__':
    model = MnasNet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)