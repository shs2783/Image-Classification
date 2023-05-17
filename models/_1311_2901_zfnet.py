'https://arxiv.org/pdf/1311.2901.pdf'

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=0.5) -> None:
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        return self.drop_out(self.act(self.linear(x)))

class ZFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=2, padding=3),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.LayerNorm(normalized_shape=[96, 55, 55]),

            ConvBlock(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=2, padding=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.LayerNorm(normalized_shape=[256, 13, 13]),

            ConvBlock(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            ConvBlock(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            ConvBlock(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.fc_layers = nn.Sequential(
            LinearBlock(256 * 6 * 6, 4096, drop_out=0.5),
            LinearBlock(4096, 4096, drop_out=0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    model = ZFNet()

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)