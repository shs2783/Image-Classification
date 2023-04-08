'https://arxiv.org/pdf/1602.07360.pdf'

import math
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3, residual_conv=False) -> None:
        super().__init__()

        self.residual_layer = ConvBlock(in_channels, e1x1+e3x3, kernel_size=1) if residual_conv else nn.Identity()
        self.sequeeze_layer = ConvBlock(in_channels, s1x1, kernel_size=1)
        self.conv1x1 = ConvBlock(s1x1, e1x1, kernel_size=1)
        self.conv3x3 = ConvBlock(s1x1, e3x3, kernel_size=3, padding=1)
        
    def forward(self, x):
        identity = self.residual_layer(x)

        x = self.sequeeze_layer(x)
        x = torch.cat([self.conv1x1(x), self.conv3x3(x)], dim=1)

        return x + identity

class SqueezeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, base_e=128, incr_e=128, freq=2, pct3x3=0.5, squeeze_ratio=0.125, num_fire_modules=8, drop_out=0.5) -> None:
        super().__init__()

        assert num_fire_modules % freq == 0, 'num_fire_modules must be divisible by freq'

        self.fist_conv_layer = ConvBlock(in_channels, out_channels=96, kernel_size=7, stride=2, padding=3)
        self.fire_modules, channels = self._create_fire_modules(base_e, incr_e, freq, pct3x3, squeeze_ratio, num_fire_modules)
        self.last_conv_layer = nn.Conv2d(in_channels=channels, out_channels=num_classes, kernel_size=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_out = nn.Dropout(p=drop_out)
    
    def _create_fire_modules(self, base_e=128, incr_e=128, freq=2, pct3x3=0.5, squeeze_ratio=0.125, num_fire_modules=8):
        layers = []
        max_pooling_idx = [0, 3, 7]

        in_channels = 96
        for i in range(num_fire_modules):
            if i in max_pooling_idx:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

            out_channels = base_e + int(incr_e * math.floor(i / freq))
            s1x1 = int(out_channels * squeeze_ratio)
            e1x1 = int(out_channels * pct3x3)
            e3x3 = int(out_channels * (1 - pct3x3))
            
            residual_conv = (i % freq == 0)
            depth = freq

            layers += [FireModule(in_channels, s1x1, e1x1, e3x3, residual_conv)]
            layers += [FireModule(out_channels, s1x1, e1x1, e3x3, residual_conv) for _ in range(depth-1)]
            in_channels = out_channels

        return nn.Sequential(*layers), out_channels
    
    def forward(self, x):
        x = self.fist_conv_layer(x)
        x = self.fire_modules(x)
        x = self.drop_out(x)

        x = self.last_conv_layer(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        return x
    
if __name__=='__main__':
    # TODO 1: applying Deep Compression with 6-bit quantization and 33% sparsity

    model = SqueezeNet()

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)