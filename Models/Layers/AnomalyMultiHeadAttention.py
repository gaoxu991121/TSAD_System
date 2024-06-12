import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyBlockList(nn.Module):
    def __init__(self, attn_list):
        super(AnomalyBlockList, self).__init__()
        self.attn_layers = nn.ModuleList(attn_list)




    def forward(self, x):
        output_list = []
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:

            output, series, prior = attn_layer(x)
            output_list.append(output)
            series_list.append(series)
            prior_list.append(prior)


        return output_list, series_list, prior_list


class AnomalyBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,window_size,drop_out_rate = 0.2 ):
        super(AnomalyBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(p=drop_out_rate)

        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.attention = AnomalyMultiHeadAttention(embed_dim, num_heads, window_size)
        self.mask = torch.triu(torch.ones(self.window_size,self.window_size),diagonal=1).to(dtype=torch.float)




    def forward(self, x):
        output, series, prior = self.attention(x,self.mask)

        output = output + x

        output = self.layer_norm1(output)

        output = self.linear1(output) + output
        output = self.dropout1(output)
        output = self.layer_norm2(output)

        return output, series, prior
class AnomalyMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,window_size ):
        super(AnomalyMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.sigma = nn.Linear(embed_dim, window_size)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)


    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear transformations
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        sigma = self.sigma(x)


        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        prior = 1/(math.sqrt(2*torch.pi) * sigma) *\
                torch.exp(-(torch.pow(self.distances,2))/(2 * sigma ) )

        # Apply scaled dot-product attention
        attention_output, series = self.scaled_dot_product_attention(query, key, value, mask)

        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out(attention_output)

        return output, series, prior

