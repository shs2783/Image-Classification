'https://arxiv.org/pdf/1406.4729.pdf'

import torch
import torch.nn as nn

spp_config = {
    # (out_channels, kernel_size, stride, padding, norm)
    # ['M', (kernel_size, stride)]

    'ZF-5': [
        (96, 7, 2, 3, True),
        ['M', (3, 2)],
        (256, 5, 2, 2, True),
        ['M', (3, 2)],
        (384, 3, 1, 1, False),
        (384, 3, 1, 1, False),
        (256, 3, 1, 1, False),
    ],

    'Convnet': [
        (96, 11, 4, 5, True),
        (256, 5, 1, 2, True),
        ['M', (3, 2)],
        (384, 3, 1, 1, False),
        ['M', (3, 2)],
        (384, 3, 1, 1, False),
        (256, 3, 1, 1, False),
    ],

    'Overfeat': [
        (96, 7, 2, 3, False),
        ['M', (3, 3)],
        (256, 5, 1, 2, False),
        ['M', (2, 2)],
        (512, 3, 1, 1, False),
        (512, 3, 1, 1, False),
        (512, 3, 1, 1, False),
        (512, 3, 1, 1, False),
        (512, 3, 1, 1, False),
    ]
}
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SPPLayer(nn.Module):
    def __init__(self, spp_output_num) -> None:
        super().__init__()

        self.pooling_layers = nn.Sequential(
            *[nn.AdaptiveAvgPool2d((k, k)) for k in spp_output_num]
        )

    def forward(self, x):
        pooling_outputs = []
        for pooling_layer in self.pooling_layers:
            pool_out = pooling_layer(x)
            pool_out = pool_out.flatten(start_dim=1)
            pooling_outputs.append(pool_out)

        output = torch.cat(pooling_outputs, dim=1)
        return output

class SPPNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, config='ZF-5', spp_output=[4, 2, 1]):
        super().__init__()
        config = spp_config[config]

        self.feature_layers, out_channels = self._create_feature_layers(config=config, in_channels=in_channels)
        self.spp_layer = SPPLayer(spp_output)

        spp_output_size = out_channels * sum([i**2 for i in spp_output])
        self.linear_layers = nn.Sequential(
            nn.Linear(spp_output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _create_feature_layers(self, config, in_channels=3):
        layers = []

        for x in config:
            if isinstance(x, tuple):
                out_channels, kernel_size, stride, padding, norm = x
                layers += [ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, norm=norm)]
                in_channels = out_channels
            elif isinstance(x, list):
                kernel_size, stride = x[1]
                layers += [nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)]

        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.spp_layer(x)
        x = self.linear_layers(x)
        return x

if __name__ == '__main__':
    model = SPPNet(in_channels=3, num_classes=1000, config='Overfeat', spp_output=[6, 3, 2, 1])
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)