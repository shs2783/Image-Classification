'https://arxiv.org/pdf/1704.04861.pdf'

import math
import torch
import torch.nn as nn

from point_depth_separable_conv import ConvBlock, SeparableConv2d

mobilenetV1_architecture = [
    # (out_channels, kernel size, stride, padding)
    (64, 3, 1, 1),
    (128, 3, 2, 1),
    (128, 3, 1, 1),
    (256, 3, 2, 1),
    (256, 3, 1, 1),
    (512, 3, 2, 1),
    [(512, 3, 1, 1), 5],
    (1024, 3, 2, 1),
    (1024, 3, 2, 4)
]

class MobileNetV1(nn.Module):
    def __init__(self, config=mobilenetV1_architecture, in_channels=3, num_classes=1000, a=1) -> None:
        super().__init__()
        self.a = a
        
        self.conv_layer = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.depthwise_blocks, out_channels = self._create_layers(config, a)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Linear(out_channels, num_classes)
    
    def _create_layers(self, architecture, a):
        layers = []
        in_channels = 32
        
        for x in architecture:
            if isinstance(x, tuple):
                out_channels, kernel_size, stride, padding = x
                out_channels = math.ceil(a * out_channels)
                layers += [
                    SeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                    ]
            
            elif isinstance(x, list):
                out_channels, kernel_size, stride, padding = x[0]
                out_channels = math.ceil(a * out_channels)
                num_repeat = x[1]
                
                layers += [
                    SeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                    for _ in range(num_repeat)
                    ]
            
            in_channels = out_channels
            
        return nn.Sequential(*layers), in_channels
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.depthwise_blocks(x)
        x = self.avg_pool(x)
        
        x = x.flatten(start_dim=1)
        x = self.fc_layer(x)
        return x
        
if __name__ == '__main__':
    a = 1  # [1, 0.75, 0.5, 0.25]
    p = 1  # [1(=224), 0.857(=192), 0.714(=160). 0.571(=128)]

    model = MobileNetV1(mobilenetV1_architecture, a=a)
    
    resolution = math.ceil(224 * p)
    x = torch.randn(1, 3, resolution, resolution)
    y = model(x)
    print(x.shape)
    print(y.shape)