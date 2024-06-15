import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow

from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa
import matplotlib.pyplot as plt
import seaborn as sns



class ATTENTION(BaseModel):
    """
    依据窗口内时间点的依赖关系进行重建
    """

    def __init__(self, config):
        super(ATTENTION, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]


        self.divice = self.config["device"]

        self.num_heads = self.config["num_heads"]

        self.attention = nn.MultiheadAttention(self.input_size, self.num_heads,batch_first=True)
        self.fc = nn.Linear(self.input_size,self.input_size)
        self.dropout = nn.Dropout(self.drop_out_rate)









    def forward(self,  x):

        batch_size, sequence_length, channels = x.size()
        #x = x.reshape(batch_size * sequence_length, channels, -1).transpose(0,1)  # [channels, batch_size * sequence_length, -1]

        attn_output, attn_weights = self.attention(x, x, x)
        #attn_output = attn_output.transpose(0, 1).reshape(batch_size,  sequence_length,channels) # [batch_size, sequence_length, channels]

        attn_output = self.dropout(self.fc(attn_output))

        return attn_output, attn_weights



    def fit(self, train_data, write_log=False):
        train_loader = self.processData(train_data)
        self.train()
        lr = self.config["learning_rate"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        # 设置余弦学习率衰减，这里的T_max是衰减周期
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)


        l = nn.MSELoss(reduction='sum')

        epoch_loss = []


        for ep in range(self.epoch):
            ep = ep + 1
            l1s = []
            running_loss = 0
            for d in train_loader:
                optimizer.zero_grad()
                item = d[0].to(self.divice)



                output,attn_weight = self.forward(item)

                loss = l(output, item)



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

        self.save()

        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": epoch_loss})

    def test(self, test_data):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_data: 测试数据

        """
        test_dataloader = self.processData(test_data)
        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.divice)


                output, attn_weight = self.forward(item)

                loss = l(output[:, -1, :], item[:, -1, :])

                score.append(loss.sum(dim=-1).detach().cpu())


            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score

    def visualize(self,test_dataloader):

        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')
        attn_weight = None
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.divice)

                output, attn_weight = self.forward(item)
                break

        print("attn hape:",attn_weight.shape)
        attn_weights_sample = attn_weight[-1]
        # 可视化注意力权重
        plt.figure(figsize=(12, 8))
        sns.heatmap(attn_weights_sample.detach().numpy(), cmap='viridis')
        plt.title("Attention Weights across Channels")
        plt.xlabel("Channels")
        plt.ylabel("Channels")
        plt.show()






