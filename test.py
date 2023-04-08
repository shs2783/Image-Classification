import sys
sys.path.append('models')

import torch
from utils import show_params
from models import MnasNet

if __name__=='__main__':
    model = MnasNet()
    show_params(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)