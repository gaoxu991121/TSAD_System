
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from Models.BaseModel import BaseModel
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import countResult, findSegment
from Utils.LogUtil import wirteLog

from Utils.ProtocolUtil import pa
import math

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm



class _ResidualBlock(nn.Module):
    def __init__(self, d_feature, num_filters, kernel_size, dilation, dropout):
        super(_ResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation  # 每个卷积核相当于 1+(k-1)*d 的大小

        self.conv1 = nn.Conv1d(
            in_channels=d_feature,
            out_channels=num_filters,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=d_feature,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = F.pad(x, (self.padding, 0))  # 在-1维度pad，左侧添加self.padding个0，右侧添加0个0，实现因果卷积
        x = self.dropout(self.relu(self.conv1(x)))
        x = F.pad(x, (self.padding, 0))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x + residual
        return x


class ResidualBlock(nn.Module):
    def __init__(self, his_len, d_feature, num_filters, kernel_size, dropout, num_layers=None):
        super(ResidualBlock, self).__init__()
        dilation_factor = kernel_size

        # 如果 num_layers 没有被传递，就计算感受野能覆盖整个输入序列的 num_layers，
        # 由论文可知感受野的计算方式为(k− 1)d，d随网络层数指数级增长
        # (kernel_size-1)*(dilation_factor**(num_layers-1))==his_len-1 => num_layers
        if num_layers is None:
            num_layers = math.ceil(
                math.log((his_len - 1) / (kernel_size - 1), dilation_factor) + 1
            )

        self.residual_blocks_list = []
        for i in range(num_layers):
            dilation = dilation_factor ** i
            res_block = _ResidualBlock(
                d_feature, num_filters, kernel_size, dilation, dropout
            )
            self.residual_blocks_list.append(res_block)
        self.residual_blocks = nn.ModuleList(self.residual_blocks_list)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        x = x.permute(0, 2, 1)
        return x



class TCNAE(BaseModel):
    """
    《An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling》

    """
    def __init__(self,config):
        super(TCNAE,self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.drop_out_rate = self.config["drop_out_rate"]
        self.output_size = self.config["input_size"]
        self.window_size = self.config["window_size"]
        self.num_filters = self.config["num_filters"]
        self.kernel_size = self.config["kernel_size"]

        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.pre_len = 1
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.fc = nn.Linear(self.hidden_size, 1)  # 1 是输出维度

        self.divice = self.config["device"]

        self.residual_blocks = ResidualBlock(self.window_size, self.input_size, self.num_filters, self.kernel_size,
                                             self.drop_out_rate)



    def forward(self,x):

        x = self.residual_blocks(x)

        return x[:, -1, :]



    def fit(self,train_data,write_log = False):
        train_loader = self.processData(train_data)
        self.train()
        lr = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        # 设置余弦学习率衰减，这里的T_max是衰减周期
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

        epoch_loss = []

        for ep in range(self.epoch):
            l1s = []
            running_loss = 0
            for d in train_loader:
                optimizer.zero_grad()
                item = d[0].to(self.divice)


                y = self.forward(item)

                loss = F.mse_loss(y, item[:,-1,:], reduction='sum')


                l1s.append(torch.mean(loss).item())
                running_loss += loss.item()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()  # 在每个epoch后更新学习率

            # 计算当前epoch的平均损失
            epoch_loss.append(running_loss / len(train_loader))

            print(f'train epoch [{ep}/{self.epoch}],\t loss = {np.mean(l1s)}')


        identifier = self.config["identifier"]
        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": epoch_loss})




    def test(self,test_data):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_data: 测试数据

        """
        test_dataloader = self.processData(test_data)

        self.eval()
        score = []
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.divice)


                y = self.forward(item)

                loss = F.mse_loss(y, item[:,-1,:], reduction='none')


                score.append(loss.sum(dim=-1).detach().cpu())

            score = torch.concatenate(score,dim=0).numpy()

            score = minMaxScaling(data = score,min_value= score.min(),max_value=score.max(),range_max=1,range_min=0)

        return score




