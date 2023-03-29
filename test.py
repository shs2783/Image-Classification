import torch
from utils import show_params
from models import ResNeXt

if __name__=='__main__':
    groups = 64
    bottleneck_width = 4
    model = ResNeXt('resnext101', groups, bottleneck_width)
    show_params(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)