import sys
sys.path.append('models')

import torch
from utils import show_params
from models import Transformer

if __name__=='__main__':
    model = Transformer()
    show_params(model)

    x = torch.randn(1, 64, 512)
    y = model(x, x)
    print(y.shape)