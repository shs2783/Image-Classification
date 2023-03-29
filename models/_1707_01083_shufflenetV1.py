'https://arxiv.org/pdf/1707.01083.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_depth_separable_conv import ConvBlock, PointwiseConv2d, DepthwiseConv2d

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, stride=1) -> None:
        super().__init__()

        self.groups = groups
        bottleneck_channels = out_channels//4

        if stride == 2:
            out_channels -= in_channels
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.avg_pool = None

        self.gconv1x1 = PointwiseConv2d(in_channels, bottleneck_channels, groups=groups)
        self.conv3x3 = DepthwiseConv2d(bottleneck_channels, bottleneck_channels, stride=stride, act=False)
        self.conv1x1 = PointwiseConv2d(bottleneck_channels, out_channels, act=False)

    def forward(self, x):
        identity = x

        x = self.gconv1x1(x)
        x = self._shuffle_channels(x)

        x = self.conv3x3(x)
        x = self.conv1x1(x)

        if self.avg_pool is not None:
            identity = self.avg_pool(identity)
            return torch.cat([identity, x], dim=1)
        else:
            return F.relu(identity + x, inplace=True)
        
    def _shuffle_channels(self, x):
        batch, channels, height, width = x.shape
        x = x.view(batch, self.groups, channels//self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.flatten(start_dim=1, end_dim=2)
        return x

class ShuffleNetV1(nn.Module):
    def __init__(self, groups=8, scale_factor=1, in_channels=3, num_classes=1000) -> None:
        super().__init__()

        out_channels = self._return_output_channels(groups, scale_factor)

        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(in_channels=24, out_channels=out_channels, groups=groups, stride=2),
            *[ResidualBlock(out_channels, out_channels, groups=groups, stride=1) for _ in range(3)]
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(in_channels=out_channels, out_channels=out_channels*2, groups=groups, stride=2),
            *[ResidualBlock(out_channels*2, out_channels*2, groups=groups, stride=1) for _ in range(7)]
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(in_channels=out_channels*2, out_channels=out_channels*4, groups=groups, stride=2),
            *[ResidualBlock(out_channels*4, out_channels*4, groups=groups, stride=1) for _ in range(3)]
        )

        out_channels = out_channels*4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layer(x)
        return x
    
    def _return_output_channels(self, groups, scale_factor=1):
        if groups == 1:
            return int(144 * scale_factor)
        if groups == 2:
            return int(200 * scale_factor)
        if groups == 3:
            return int(240 * scale_factor)
        if groups == 4:
            return int(272 * scale_factor)
        if groups == 8:
            return int(384 * scale_factor)
        
if __name__=='__main__':
    # TODO: error (out_channels must be divisible by groups) occured when groups [2, 4] with scale factor [0.25, 0.5]
    groups = 8  # [1, 2, 3, 4, 8]
    scale_factor = 1  # [0.25, 0.5, 1]
    model = ShuffleNetV1(groups, scale_factor)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)