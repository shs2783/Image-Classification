import sys
sys.path.append('models')

import torch
from utils import show_params
from models import SqueezeNet

if __name__=='__main__':
    model = SqueezeNet()
    show_params(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)