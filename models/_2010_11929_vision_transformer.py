'https://arxiv.org/pdf/2010.11929v2.pdf'

import torch
import torch.nn as nn
from transformer import Encoder

class ViT(nn.Module):
    def __init__(self, patch_size=16, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.linear_project = nn.Linear(patch_size*patch_size, d_model)
        self.encoder = Encoder(d_model, num_head, d_ff, num_repeats, drop_out)

    def forward(self, x):
        x = self.get_patches(x)
        x = self.linear_project(x)
        x = self.encoder(x)
        return x

    def get_patches(self, x):
        batch, channels, height, width = x.shape
        patch_size = self.patch_size
        num_patches = height // patch_size
        
        x = x.view(batch, channels, num_patches, patch_size, num_patches, patch_size)
        x = x.transpose(3, 4).contiguous()
        x = x.view(batch, channels*num_patches*num_patches, patch_size*patch_size)
        return x
    
if __name__ == '__main__':
    model = ViT()
    
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)