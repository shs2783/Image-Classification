'https://arxiv.org/pdf/1905.11946.pdf'

import math
import torch
import torch.nn as nn

efficientnet_architecture = [
    # (out_channels, kernel_size, stride, padding, expansion, num_repeats)
    (16, 3, 1, 1, 1, 1),
    (24, 3, 2, 1, 6, 2),
    (40, 5, 2, 2, 6, 2),
    (80, 3, 2, 1, 6, 3),
    (112, 5, 1, 2, 6, 3),
    (192, 5, 2, 2, 6, 4),
    (320, 3, 1, 1, 6, 1),
]

class ConvBlock(nn.Module):  # use SiLU
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PointwiseConv2d(ConvBlock): # conv1x1 (= BottleNeck layer)
    def __init__(self, in_channels, out_channels, kernel_size=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

class DepthwiseConv2d(ConvBlock):  # conv3x3
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, groups=out_channels, **kwargs)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16) -> None:
        super().__init__()

        reduction_channels = math.ceil(channels / reduction_ratio)
        self.squeeze_excitation_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, reduction_channels, kernel_size=1),  # same as nn.Linear(gate_channels, channels),
            nn.ReLU(),
            nn.Conv2d(reduction_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.squeeze_excitation_layers(x)
    
class MBConvBlock(nn.Module):  # InvertedResidual + SE
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expansion=6, reduction_ratio=16) -> None:
        super().__init__()

        self.residual = (in_channels == out_channels)
        hidden_channels = expansion * in_channels

        self.bottleneck_layer = nn.Sequential(
            PointwiseConv2d(in_channels, hidden_channels),
            DepthwiseConv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding),
            PointwiseConv2d(hidden_channels, out_channels),
            SEBlock(out_channels, reduction_ratio),
        )

    def forward(self, x):
        identity = x
        scaled_feature = self.bottleneck_layer(x)
        
        if self.residual:
            return identity + scaled_feature
        else:
            return scaled_feature
    
class EfficientNet(nn.Module):
    # TODO 1: stochastic depth with survival probability 0.8
    # TODO 2: MnasNet ConvBlock

    def __init__(self, config=efficientnet_architecture, depth_multiplier=1.2, width_multiplier=1.1, reduction_ratio=16, in_channels=3, num_classes=1000) -> None:
        super().__init__()

        self.conv1_channels = conv1_channels = math.ceil(32 * width_multiplier)
        self.conv2_channels = conv2_channels = math.ceil(1280 * width_multiplier)

        self.conv_layer1 = ConvBlock(in_channels=in_channels, out_channels=conv1_channels, kernel_size=3, stride=2, padding=1)
        self.mobile_se_conv_layers, out_channels = self._create_layers(config, reduction_ratio, depth_multiplier, width_multiplier)
        self.conv_layer2 = ConvBlock(in_channels=out_channels, out_channels=conv2_channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_layer3 = nn.Conv2d(in_channels=conv2_channels, out_channels=num_classes, kernel_size=1)

    def _create_layers(self, architecture, reduction_ratio=16, depth_multiplier=1.2, width_multiplier=1.1):
        layers = []
        in_channels = self.conv1_channels
        
        for x in architecture:
            out_channels, kernel_size, stride, padding, expansion, num_repeat = x
            num_repeat = int(num_repeat * depth_multiplier)
            out_channels = math.ceil(out_channels * width_multiplier)

            layers += [MBConvBlock(in_channels, out_channels, kernel_size, stride, padding, expansion, reduction_ratio)]
            layers += [MBConvBlock(out_channels, out_channels, kernel_size, 1, padding, expansion, reduction_ratio) for _ in range(num_repeat-1)]
            in_channels = out_channels
            
        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.mobile_se_conv_layers(x)
        x = self.conv_layer2(x)
        x = self.avg_pool(x)
        x = self.conv_layer3(x)
        x = x.flatten(start_dim=1)
        return x

if __name__ == '__main__':
    compound_coefficient = 0
    depth_multiplier = 1.2 ** compound_coefficient
    width_multiplier = 1.1 ** compound_coefficient
    resolution_multi = 1.15 ** compound_coefficient
    
    model = EfficientNet(depth_multiplier=depth_multiplier, width_multiplier=width_multiplier, reduction_ratio=16)

    resolution = int(resolution_multi * 224)
    x = torch.randn(1, 3, resolution, resolution)
    y = model(x)
    print(y.shape)