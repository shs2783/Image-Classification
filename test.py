import sys
sys.path.append('models')

import torch
from utils import show_params
from models import EfficientNet

if __name__=='__main__':
    compound_coefficient = 0
    depth_multiplier = 1.2 ** compound_coefficient
    width_multiplier = 1.1 ** compound_coefficient
    resolution_multi = 1.15 ** compound_coefficient
    
    model = EfficientNet(depth_multiplier=depth_multiplier, width_multiplier=width_multiplier, reduction_ratio=16)
    show_params(model)

    resolution = int(resolution_multi * 224)
    x = torch.randn(1, 3, resolution, resolution)
    y = model(x)
    print(y.shape)