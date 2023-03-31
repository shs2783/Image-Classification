'https://arxiv.org/pdf/1801.04381.pdf'

import torch
import torch.nn as nn

mobilenetV2_architecture = [
    # (out_channels, stride, expansion, num_repeats)
    (16, 1, 1, 1),
    (24, 2, 6, 2),
    (32, 2, 6, 3),
    (64, 2, 6, 4),
    (96, 1, 6, 3),
    (160, 2, 6, 3),
    (320, 1, 6, 1),
]

class ConvBlock(nn.Module):  # use relu6
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU6() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class PointwiseConv2d(ConvBlock): # conv1x1 (= BottleNeck layer)
    def __init__(self, in_channels, out_channels, kernel_size=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

class DepthwiseConv2d(ConvBlock):  # conv3x3
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, groups=out_channels, **kwargs)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6) -> None:
        super().__init__()

        self.residual = (in_channels == out_channels)
        hidden_channels = expansion * in_channels
        act = (stride == 1)

        self.bottleneck_layers = nn.Sequential(
            PointwiseConv2d(in_channels, hidden_channels),
            DepthwiseConv2d(hidden_channels, hidden_channels, stride=stride),
            PointwiseConv2d(hidden_channels, out_channels, act=act)
        )

    def forward(self, x):
        if self.residual:
            return x + self.bottleneck_layers(x)
        else:
            return self.bottleneck_layers(x)


class MobileNetV2(nn.Module):
    def __init__(self, config=mobilenetV2_architecture, in_channels=3, num_classes=1000) -> None:
        super().__init__()
        
        self.conv_layer1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bottleneck_residual_layers = self._create_layers(config)
        self.conv_layer2 = ConvBlock(in_channels=320, out_channels=1280, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_layer3 = nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1)
    
    def _create_layers(self, architecture):
        layers = []
        in_channels = 32
        
        for x in architecture:
            out_channels, stride, expansion, num_repeat = x
            layers += [InvertedResidualBlock(in_channels, out_channels, stride=stride, expansion=expansion)]
            layers += [InvertedResidualBlock(out_channels, out_channels, stride=1, expansion=expansion) for _ in range(num_repeat-1)]
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.bottleneck_residual_layers(x)
        x = self.conv_layer2(x)
        x = self.avg_pool(x)
        x = self.conv_layer3(x)
        x = x.flatten(start_dim=1)
        return x
        
if __name__ == '__main__':
    model = MobileNetV2(mobilenetV2_architecture)
    
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)