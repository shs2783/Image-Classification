'https://arxiv.org/pdf/1610.02357.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_depth_separable_conv import ConvBlock, SeparableConv2d

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=2)
        self.separable_conv_layer1 = SeparableConv2d(in_channels, hidden_channels, act=False)
        self.separable_conv_layer2 = SeparableConv2d(hidden_channels, out_channels, act=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        residual = self.conv(x)

        x = F.relu(x, inplace=True)
        x = self.separable_conv_layer1(x)

        x = F.relu(x, inplace=True)
        x = self.separable_conv_layer2(x)
        
        x = self.max_pool(x)
        return x + residual
        
class EntryFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.separable_conv_block1 = SeparableConvBlock(in_channels=64, hidden_channels=128, out_channels=128)
        self.separable_conv_block2 = SeparableConvBlock(in_channels=128, hidden_channels=256, out_channels=256)
        self.separable_conv_block3 = SeparableConvBlock(in_channels=256, hidden_channels=728, out_channels=728)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.separable_conv_block1(x)
        x = self.separable_conv_block2(x)
        x = self.separable_conv_block3(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, in_out_channels=728) -> None:
        super().__init__()
        
        self.separable_conv_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(in_out_channels, in_out_channels, act=False),
            nn.ReLU(inplace=True),
            SeparableConv2d(in_out_channels, in_out_channels, act=False),
            nn.ReLU(inplace=True),
            SeparableConv2d(in_out_channels, in_out_channels, act=False),
        )
    
    def forward(self, x):
        identity = x
        x = self.separable_conv_layer(x)
        return identity + x

class MiddleFlow(nn.Module):
    def __init__(self, in_out_channels=728, block_repeat_num=8) -> None:
        super().__init__()
        
        self.separable_conv_block = nn.Sequential(
            *[ MiddleBlock(in_out_channels) for _ in range(block_repeat_num) ]
        )
    
    def forward(self, x):
        return self.separable_conv_block(x)
    
class ExitFlow(nn.Module):
    def __init__(self, num_class=1000) -> None:
        super().__init__()
        
        self.separable_conv_block = SeparableConvBlock(in_channels=728, hidden_channels=728, out_channels=1024)
        self.separable_conv1 = SeparableConv2d(in_channels=1024, out_channels=1536, act=False)
        self.separable_conv2 = SeparableConv2d(in_channels=1536, out_channels=2048, act=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, num_class)
        
    def forward(self, x):
        x = self.separable_conv_block(x)
        
        x = F.relu(self.separable_conv1(x), inplace=True)
        x = F.relu(self.separable_conv2(x), inplace=True)
        x = self.global_avg_pool(x)
        
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x

class XceptionNet(nn.Module):
    def __init__(self, middle_repeat_num=8) -> None:
        super().__init__()
        self.middle_repeat_num = middle_repeat_num

        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()
        
    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x
    
if __name__ == '__main__':
    model = XceptionNet()

    x = torch.randn((1, 3, 299, 299))
    y = model(x)
    print(y.shape)
    