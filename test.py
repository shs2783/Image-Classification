import torch
from utils import show_params
from models import ShuffleNetV1

if __name__=='__main__':
    model = ShuffleNetV1(groups=3, scale_factor=1)
    show_params(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)