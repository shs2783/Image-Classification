'https://arxiv.org/pdf/1409.1556.pdf'

import torch
import torch.nn as nn

vgg_architectures = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg11-LRN': [64, 'LRN', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
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
    
class VggNet(nn.Module):
    def __init__(self, config='vgg16', in_channels=3, num_classes=1000):
        ''' config: vgg11, vgg13, vgg16, vgg19, vgg11-LRN '''
        
        super().__init__()

        architecture = vgg_architectures[config]
        self.conv_layers = self.create_conv_layers(architecture, in_channels)
        self.fc_layers = nn.Sequential(
            LinearBlock(7*7*512, 4096, drop_out=0.5),
            LinearBlock(4096, 4096, drop_out=0.5),
            nn.Linear(4096, num_classes)
        )
        
    def create_conv_layers(self, architecture, in_channels):
        layers = []
        
        for x in architecture:
            if isinstance(x, int):
                out_channels = x
                layers += [ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)]
                in_channels = x
            
            elif isinstance(x, str):
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
                elif x == 'LRN':
                    layers += [nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layers(x)        
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x
    
if __name__ == '__main__':
    model = VggNet('vgg16')

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)