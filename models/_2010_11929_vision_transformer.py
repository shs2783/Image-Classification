'https://arxiv.org/pdf/2010.11929v2.pdf'

import torch
import torch.nn as nn
from transformer import MultiHeadAttention
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, drop_out=0.1) -> None:
        super().__init__()

        self.fc_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=drop_out),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=drop_out)
        )
    
    def forward(self, x):
        return self.fc_layer(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, drop_out=0.1) -> None:
        super().__init__()

        self.self_attention_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadAttention(d_model, num_head)
        )

        self.fc_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionWiseFeedForward(d_model, d_ff, drop_out)
        )

    def forward(self, x):
        enc_attention = self.self_attention_layer(x)
        enc_attention = x + enc_attention

        enc_output = self.fc_layer(enc_attention)
        enc_output = enc_attention + enc_output
        return enc_output
    

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1) -> None:
        super().__init__()

        self.layers = [EncoderBlock(d_model, num_head, d_ff, drop_out) for _ in range(num_repeats)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, 
                in_channels=3, 
                num_classes=1000, 
                img_size=224, 
                patch_size=16, 
                d_model=512, 
                num_head=8, 
                d_ff=2048, 
                num_repeats=6, 
                drop_out=0.1, 
                pre_training=False) -> None:
        
        super().__init__()

        self.patch_size = patch_size
        self.pre_training = pre_training
        num_patches = (img_size // patch_size) ** 2

        self.linear_project = nn.Sequential(
            nn.LayerNorm(in_channels*patch_size*patch_size),
            nn.Linear(in_channels*patch_size*patch_size, d_model)
        )

        self.cls_token_parameter = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embeddings_parameter = nn.Parameter(torch.randn(1, num_patches+1, d_model))
        self.transformer_encoder = Encoder(d_model, num_head, d_ff, num_repeats, drop_out)
        self.drop_out = nn.Dropout(p=drop_out)

        if pre_training:
            self.hidden_fc_layer = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Dropout(p=drop_out)
            )

        self.fc_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape = (batch, channels, height, width)
        img_patches = self.get_patches(x)  # (batch, num_patches, channels*patch_size*patch_size)
        patch_embeddings = self.linear_project(img_patches)  # (batch, num_patches, d_model)
        batch, num_patches, d_model = patch_embeddings.shape

        cls_tokens = self.cls_token_parameter.repeat(batch, 1, 1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (batch, num_patches+1, d_model)
        x += self.position_embeddings_parameter  # (batch, num_patches+1, d_model)
        x = self.drop_out(x)

        x = self.transformer_encoder(x)  # (batch, num_patches+1, d_model)
        x = x.mean(dim=1)  # (batch, d_model)

        if self.pre_training:
            x = self.hidden_fc_layer(x)  # (batch, d_model)
        x = self.fc_layer(x)  # (batch, num_classes)
        return x

    def get_patches(self, x):
        batch, channels, img_height, img_width = x.shape
        patch_size = self.patch_size
        patch_num = img_height // patch_size  # or img_width // patch_size 
        
        x = x.view(batch, channels, patch_num, patch_size, patch_num, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (batch, patch_num, patch_num, channels, patch_size, patch_size)
        x = x.view(batch, patch_num*patch_num, channels*patch_size*patch_size)
        return x
    
if __name__ == '__main__':
    model = ViT()
    
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)