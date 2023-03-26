'https://arxiv.org/pdf/1801.04381.pdf'

import math
import torch
import torch.nn as nn

from .point_and_depth_wise_conv import ConvBlock, DepthwiseSeparableConv2d

class MobileNetV2(nn.Module):
    ...