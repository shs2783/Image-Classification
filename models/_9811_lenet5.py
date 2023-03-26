'http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf'

import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=1)
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x
    
if __name__ == '__main__':
    x = torch.randn(16, 1, 32, 32)
    model = LeNet()
    output = model(x)
    print(output.shape)