'https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf'


import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=0.5) -> None:
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        return self.drop_out(self.act(self.linear(x)))

class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4, padding=0, norm=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            ConvBlock(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2, norm=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            ConvBlock(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1, norm=False),
            ConvBlock(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1, norm=False),
            ConvBlock(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, norm=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.fc_layers = nn.Sequential(
            LinearBlock(256*6*6, 4096, drop_out=0.5),
            LinearBlock(4096, 4096, drop_out=0.5),
            nn.Linear(4096, 1000)
        )

        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    model = AlexNet()
    
    x = torch.randn(1, 3, 227, 227)
    y = model(x)
    print(y.shape)