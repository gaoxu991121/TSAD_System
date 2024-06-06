import torch
import torch.nn as nn
import math


class PE(nn.Module):
    def __init__(self, d_model,dropout = 0.1, max_len=5000):
        super(PE, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))

        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, d_model)
        # 将位置编码加到输入张量中
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)