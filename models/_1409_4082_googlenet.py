'https://arxiv.org/pdf/1409.4842.pdf'

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))
    
class Conv1x1(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, padding=0)

class Conv3x3(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1)

class Conv5x5(ConvBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=5, padding=2)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj) -> None:
        super().__init__()

        self.branch1 = Conv1x1(in_channels, out_1x1)
        
        self.branch2 = nn.Sequential(
            Conv1x1(in_channels, reduce_3x3),
            Conv3x3(reduce_3x3, out_3x3)
        )

        self.branch3 = nn.Sequential(
            Conv1x1(in_channels, reduce_5x5),
            Conv5x5(reduce_5x5, out_5x5)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv1x1(in_channels, pool_proj)
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1000) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=1),

            nn.Flatten(start_dim=1),
            nn.Linear(4*4*128, 1024),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, is_train=True) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, k=2),

            ConvBlock(in_channels=64, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, k=2),
        )

        self.inception_3a = InceptionBlock(in_channels=192, out_1x1=64, reduce_3x3=96, out_3x3=128, reduce_5x5=16, out_5x5=32, pool_proj=32)
        self.inception_3b = InceptionBlock(in_channels=256, out_1x1=128, reduce_3x3=128, out_3x3=192, reduce_5x5=32, out_5x5=96, pool_proj=64)

        self.inception_4a = InceptionBlock(in_channels=480, out_1x1=192, reduce_3x3=96, out_3x3=208, reduce_5x5=16, out_5x5=48, pool_proj=64)
        self.inception_4b = InceptionBlock(in_channels=512, out_1x1=160, reduce_3x3=112, out_3x3=224, reduce_5x5=24, out_5x5=64, pool_proj=64)
        self.inception_4c = InceptionBlock(in_channels=512, out_1x1=128, reduce_3x3=128, out_3x3=256, reduce_5x5=24, out_5x5=64, pool_proj=64)
        self.inception_4d = InceptionBlock(in_channels=512, out_1x1=112, reduce_3x3=144, out_3x3=288, reduce_5x5=32, out_5x5=64, pool_proj=64)
        self.inception_4e = InceptionBlock(in_channels=528, out_1x1=256, reduce_3x3=160, out_3x3=320, reduce_5x5=32, out_5x5=128, pool_proj=128)

        self.inception_5a = InceptionBlock(in_channels=832, out_1x1=256, reduce_3x3=160, out_3x3=320, reduce_5x5=32, out_5x5=128, pool_proj=128)
        self.inception_5b = InceptionBlock(in_channels=832, out_1x1=384, reduce_3x3=192, out_3x3=384, reduce_5x5=48, out_5x5=128, pool_proj=128)

        self.aux_classifier1 = AuxiliaryClassifier(512, num_classes) if is_train else None
        self.aux_classifier2 = AuxiliaryClassifier(528, num_classes) if is_train else None

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

        self.is_train = is_train

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.max_pool(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool(x)
        
        x = self.inception_4a(x)
        aux1 = x
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux2 = x
        x = self.inception_4e(x)
        x = self.max_pool(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc_layer(x)

        if self.is_train:
            aux1 = self.aux_classifier1(aux1)
            aux2 = self.aux_classifier2(aux2)
            return x, aux1, aux2
        else:
            return x
        

if __name__ == '__main__':
    model = GoogleNet(is_train=True)
    
    x = torch.randn(1, 3, 224, 224)
    y, aux1, aux2 = model(x)
    print(y.shape, aux2.shape, aux1.shape)