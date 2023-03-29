'https://arxiv.org/pdf/1611.05431.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

resnext_config = {
    'resnext18': [[2, 2, 2, 2], False],
    'resnext34': [[3, 4, 6, 3], False],
    'resnext50': [[3, 4, 6, 3], True],
    'resnext101': [[3, 4, 23, 3], True],
    'resnext152': [[3, 8, 36, 3], True]
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=32, bottleneck_width=4, bottleneck=True, down_sample=None) -> None:
        super().__init__()
        self.down_sample = down_sample
        width = bottleneck_width * groups

        if bottleneck:
            self.res_layers = nn.Sequential(
                ConvBlock(in_channels, width, kernel_size=1, stride=1, padding=0, act=True),
                ConvBlock(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, act=True),
                ConvBlock(width, out_channels, kernel_size=1, stride=1, padding=0, act=False),
            )

        else:
            self.res_layers = nn.Sequential(
                ConvBlock(in_channels, width, kernel_size=3, stride=1, padding=1, groups=groups, act=True),
                ConvBlock(width, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, act=False)
            )
            
    def forward(self, x):
        identity = self.down_sample(x) if self.down_sample else x
        x = self.res_layers(x)
        return F.relu(x + identity, inplace=True)
    

class ResNeXt(nn.Module):
    def __init__(self, config='resnext50', groups=32, bottleneck_width=4, in_channels=3, num_classes=1000) -> None:
        ''' config: resnext18, resnext34, resnext50, resnext101, resnext152 '''
        
        super().__init__()
        model_config = resnext_config[config]
        num_repeats = model_config[0]
        bottleneck = model_config[1]

        self.groups = groups
        self.bottleneck_width = bottleneck_width
        self.res_channels = 64

        self.conv_layer = ConvBlock(in_channels, out_channels=self.res_channels, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layer1 = self._make_residual_layers(num_repeats[0], bottleneck, down_sample=False)
        self.res_layer2 = self._make_residual_layers(num_repeats[1], bottleneck, down_sample=True)
        self.res_layer3 = self._make_residual_layers(num_repeats[2], bottleneck, down_sample=True)
        self.res_layer4 = self._make_residual_layers(num_repeats[3], bottleneck, down_sample=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if bottleneck:
            self.fc_layer = nn.Linear(2048, num_classes)
        else:
            self.fc_layer = nn.Linear(512, num_classes)

    def _make_residual_layers(self, num_repeat, bottleneck, down_sample):
        layers = []
        out_channels = self.res_channels*4 if bottleneck else self.res_channels
        groups = self.groups
        bottleneck_width = self.bottleneck_width
            
        if down_sample:
            stride = 2
            in_channels = out_channels//2
            down_sample = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            stride = 1
            in_channels = self.res_channels
            down_sample = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride)
        
        layers += [ResidualBlock(in_channels, out_channels, stride, groups, bottleneck_width, bottleneck, down_sample)]
        for _ in range(num_repeat-1):
            layers += [ResidualBlock(out_channels, out_channels, 1, groups, bottleneck_width, bottleneck, down_sample=None)]

        self.res_channels *= 2
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.max_pool(x)

        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.avg_pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc_layer(x)
        return x
    
if __name__=='__main__':
    groups = 32
    bottleneck_width = 8
    model = ResNeXt('resnext101', groups, bottleneck_width)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)