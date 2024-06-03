

import torch
import torch.nn as nn




class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

    def shuffle(self,data):
        # 假设你的张量数据
        (batch_size,seq_len,channels) = data.shape
        # 生成随机排列的索引
        indices = torch.randperm(batch_size)

        # 对时间步维度进行打乱
        data = data[indices]

        return data

