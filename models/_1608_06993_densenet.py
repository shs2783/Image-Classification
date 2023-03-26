'https://arxiv.org/pdf/1608.06993.pdf'

import torch
import torch.nn as nn

densnet_config = {
    'densenet121': [6, 12, 24, 16],
    'densenet169': [6, 12, 32, 32],
    'densenet201': [6, 12, 48, 32],
    'densenet264': [6, 12, 64, 48],
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avg_pool(self.conv(x))
    
class DenseBlock(nn.Module):
    def __init__(self, k0, growth_rate, num_repeat) -> None:
        super().__init__()

        self.dense_layer = nn.ModuleList()
        for i in range(num_repeat):
            in_channels = k0 + growth_rate*i
            out_channels = growth_rate

            self.dense_layer.append(
                nn.Sequential(
                    ConvBlock(in_channels, growth_rate*4, kernel_size=1),
                    ConvBlock(growth_rate*4, out_channels, kernel_size=3, padding=1)
                )
            )
            
    def forward(self, x):
        dense_feature_map = x
        for layer in self.dense_layer:
            x = layer(dense_feature_map)
            dense_feature_map = torch.cat([dense_feature_map, x], dim=1)

        return dense_feature_map

class DenseNet(nn.Module):
    def __init__(self, config='densenet121', in_channels=3, init_features=64, num_classes=1000, growth_rate=32, compression_factor=0.5) -> None:
        ''' config: densenet121, densenet169, densenet201, densenet264 '''
        
        super().__init__()

        num_repeats = densnet_config[config]
        self.conv_layer = ConvBlock(in_channels=in_channels, out_channels=init_features, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_transition_layers, out_features = self._make_dense_transition_layers(num_repeats, init_features, growth_rate, compression_factor)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Linear(out_features, num_classes)

    def _make_dense_transition_layers(self, num_repeats, init_features, growth_rate, compression_factor):
        layers = []
        k0 = init_features

        for i, num_repeat in enumerate(num_repeats):
            layers += [DenseBlock(k0, growth_rate, num_repeat)]
            k0 = k0 + growth_rate * num_repeat

            if i < len(num_repeats) - 1:
                layers += [TransitionLayer(k0, int(k0*compression_factor))]
                k0 = int(k0*compression_factor)

        return nn.Sequential(*layers), k0

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.max_pool(x)

        x = self.dense_transition_layers(x)
        x = self.avg_pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc_layer(x)
        return x
    
if __name__=='__main__': 
    model = DenseNet('densenet121')

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)