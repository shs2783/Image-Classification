import torch
from utils import show_params
from models import ResNet

if __name__=='__main__':
    model = ResNet('resnet101')
    show_params(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)