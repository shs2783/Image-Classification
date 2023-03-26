'https://arxiv.org/pdf/1610.02357.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_and_depth_wise_conv import DepthwiseSeparableConv2d, ConvBlock

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.separable_conv1 = DepthwiseSeparableConv2d(in_channels, hidden_channels)
        self.separable_conv2 = DepthwiseSeparableConv2d(hidden_channels, out_channels, act=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        residual = self.conv(x)
        x = F.relu(self.separable_conv1(x), inplace=True)
        x = self.max_pool(self.separable_conv2(x))
        
        return F.relu(x + residual)
        
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
    def __init__(self, in_out_channels=728, layer_repeat_num=3) -> None:
        super().__init__()
        
        self.separable_conv = nn.Sequential(
            [DepthwiseSeparableConv2d(in_out_channels, in_out_channels, act=False)
                for _ in range(layer_repeat_num)]
        )
    
    def forward(self, x):
        identity = x
        x = self.separable_conv(x)

        return F.relu(identity + x)

class MiddleFlow(nn.Module):
    def __init__(self, in_out_channels=728, block_repeat_num=8, layer_repeat_num=3) -> None:
        super().__init__()
        
        self.separable_conv_block = nn.Sequential(
            *[ MiddleBlock(in_out_channels, layer_repeat_num) for _ in range(block_repeat_num) ]
        )
    
    def forward(self, x):
        return self.separable_conv_block(x)
    
class ExitFlow(nn.Module):
    def __init__(self, num_class=1000, feature_extraction = False) -> None:
        super().__init__()
        self.feature_extraction = feature_extraction
        
        self.separable_conv_block = SeparableConvBlock(in_channels=728, hidden_channels=728, out_channels=1024)
        self.separable_conv1 = DepthwiseSeparableConv2d(in_channels=1024, out_channels=1536)
        self.separable_conv2 = DepthwiseSeparableConv2d(in_channels=1536, out_channels=2048)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, num_class)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(x)
        x = self.separable_conv_block(x)
        
        x = self.relu(self.separable_conv1(x))
        x = self.relu(self.separable_conv2(x))
        x = self.global_avg_pool(x)
        if self.feature_extraction:
            return x
        
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x

class XceptionNet(nn.Module):
    def __init__(self, middle_repeat_num=8) -> None:
        super().__init__()
        self.middle_repeat_num = middle_repeat_num
        self.feature_extraction = False
        
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()
        
    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x, self.feature_extraction)
        return x
    
if __name__ == '__main__':
    model = XceptionNet()
    
    x = torch.randn((1, 3, 299, 299))
    y = model(x)
    print(y.shape)
    