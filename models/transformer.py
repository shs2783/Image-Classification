'https://arxiv.org/pdf/1706.03762.pdf'
'https://github.com/hyunwoongko/transformer/'
'https://cpm0722.github.io/pytorch-implementation/transformer'

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64) -> None:
        super().__init__()

        self.scaling_factor = d_k ** 0.5

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, num_head, seq_len, d_k)

        key = key.transpose(-2, -1)  # (batch_size, num_head, d_k, seq_len)
        attention_score = torch.matmul(query, key) / self.scaling_factor

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e+8)

        attention_weight = F.softmax(attention_score , dim=-1)
        attention = torch.matmul(attention_weight, value)

        return attention, attention_weight
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8) -> None:
        super().__init__()

        d_k = int(d_model / num_head)
        d_v = int(d_model / num_head)

        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v

        self.query_linear = nn.Linear(d_model, d_k*num_head)
        self.key_linear = nn.Linear(d_model, d_k*num_head)
        self.value_linear = nn.Linear(d_model, d_v*num_head)

        self.attention_layers = ScaledDotProductAttention(d_k)
        self.fc_layer = nn.Linear(d_v*num_head, d_model)

    def forward(self, x, x2=None, mask=None):
        batch, seq_len, dimension = x.shape

        if x2 is not None:
            query = x
            key = x2
            value = x2
        else:
            query = key = value = x
        
        query = self.query_linear(query)  # (batch, seq_len, dimension)  ## dimension = d_model
        key = self.key_linear(key)        # (batch, seq_len, dimension)
        value = self.value_linear(value)  # (batch, seq_len, dimension)

        query = self._split_dimension(query)  # (batch, num_head, seq_len, dimension//num_head)
        key = self._split_dimension(key)      # (batch, num_head, seq_len, dimension//num_head)
        value = self._split_dimension(value)  # (batch, num_head, seq_len, dimension//num_head)

        attention, attention_weights = self.attention_layers(query, key, value, mask)  # (batch, num_head, seq_len, dimension//num_head)
        attention = attention.transpose(1, 2).contiguous()  # (batch, seq_len, num_head, dimension//num_head)
        attention = attention.view(batch, seq_len, dimension)
        output = self.fc_layer(attention)
        return output

    def _split_dimension(self, x):
        batch, seq_len, dimension = x.shape
        num_head = self.num_head

        x = x.view(batch, seq_len, num_head, dimension//num_head)
        x = x.transpose(1, 2)  # (batch, num_head, seq_len, dimension//num_head)
        return x
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, drop_out=0.1) -> None:
        super().__init__()

        self.fc_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=drop_out)
        )
    
    def forward(self, x):
        return self.fc_layer(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, drop_out=0.1) -> None:
        super().__init__()

        self.self_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.fc_layer = PositionWiseFeedForward(d_model, d_ff, drop_out)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, enc_mask=None):
        enc_attention = self.self_attention_layer(x, mask=enc_mask)
        enc_attention = self.layer_norm1(x + enc_attention)

        enc_output = self.fc_layer(enc_attention)
        enc_output = self.layer_norm2(enc_attention + enc_output)
        return enc_output

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, drop_out=0.1) -> None:
        super().__init__()

        self.self_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.enc_dec_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.enc_dec_fc_layer = PositionWiseFeedForward(d_model, d_ff, drop_out)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, dec_mask=None, enc_dec_mask=None):
        dec_attention = self.self_attention_layer(x, mask=dec_mask)
        dec_attention = self.layer_norm1(x + dec_attention)

        enc_dec_attention = self.enc_dec_attention_layer(dec_attention, enc_output, enc_dec_mask)
        enc_dec_attention = self.layer_norm2(dec_attention + enc_dec_attention)

        enc_dec_output = self.enc_dec_fc_layer(enc_dec_attention)
        enc_dec_output = self.layer_norm3(enc_dec_attention + enc_dec_output)
        return enc_dec_output
    
    
class Encoder(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1) -> None:
        super().__init__()

        self.layers = [EncoderBlock(d_model, num_head, d_ff, drop_out) for _ in range(num_repeats)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, enc_mask=None):
        for layer in self.layers:
            x = layer(x, enc_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1) -> None:
        super().__init__()

        self.layers = [DecoderBlock(d_model, num_head, d_ff, drop_out) for _ in range(num_repeats)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, enc_output, dec_mask=None, enc_dec_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, dec_mask, enc_dec_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1) -> None:
        super().__init__()
        
        self.token_embedding = nn.Linear(1, d_model)
        self.positional_encoding = nn.Linear(d_model, d_model)

        self.encoder = Encoder(d_model, num_head, d_ff, num_repeats, drop_out)
        self.decoder = Decoder(d_model, num_head, d_ff, num_repeats, drop_out)
    
    def forward(self, enc_x, dec_x):
        # enc_x, dec_x shape = (batch, seq_len)
        enc_mask = self._make_pad_mask(enc_x, enc_x)  # (batch, 1, seq_len, seq_len)
        enc_dec_mask = self._make_pad_mask(dec_x, enc_x)  # (batch, 1, seq_len, seq_len)
        dec_mask =  self._make_pad_mask(dec_x, dec_x) & self._make_subsequent_mask(dec_x, dec_x)  # (batch, 1, seq_len, seq_len)

        enc_x = self.token_embedding(enc_x.unsqueeze(-1))  # (batch, seq_len, d_model)
        dec_x = self.token_embedding(dec_x.unsqueeze(-1))  # (batch, seq_len, d_model)

        enc_x = self.positional_encoding(enc_x)  # (batch, seq_len, d_model)
        dec_x = self.positional_encoding(dec_x)  # (batch, seq_len, d_model)

        enc_output = self.encoder(enc_x, enc_mask)
        dec_output = self.decoder(dec_x, enc_output, dec_mask, enc_dec_mask)
        return dec_output

    def _make_pad_mask(self, query, key, query_pad_idx=0, key_pad_idx=0):
        batch, len_q = query.shape
        batch, len_k = key.shape

        query_mask = (query != query_pad_idx).view(batch, 1, len_q, 1)
        query_mask = query_mask.repeat(1, 1, 1, len_k)  # (batch, 1, len_q, len_k)

        key_mask = (key != key_pad_idx).view(batch, 1, 1, len_k)
        key_mask = key_mask.repeat(1, 1, len_q, 1)  # (batch, 1, len_q, len_k)

        pad_mask = query_mask & key_mask
        return pad_mask

    def _make_subsequent_mask(self, query, key):
        batch, len_q = query.shape
        batch, len_k = key.shape

        matrix = torch.ones(len_q, len_k)
        subsequent_mask = torch.tril(matrix).bool()  # lower triangular matrix
        return subsequent_mask

if __name__ == '__main__':
    # TODO 1: token embedding
    # TODO 2: positional encoding

    model = Transformer(d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1)

    x = torch.randn(1, 64)  # (batch, seq_len, dimension)
    y = model(x, x)
    print(y.shape)