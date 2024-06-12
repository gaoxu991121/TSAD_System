import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.ChannelWiseAttention import ChannelWiseAttention
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Models.Layers.SAM import SAM
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow

from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa
import matplotlib.pyplot as plt
import seaborn as sns



class CHANNELATTENTION(BaseModel):
    """
    依据通道的依赖关系进行重建
    """

    def __init__(self, config):
        super(CHANNELATTENTION, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]


        self.device = self.config["device"]

        self.mask = self.config["mask"]
        self.predict_length = self.config["predict_length"]

        self.attention = ChannelWiseAttention(input_size=self.input_size,seq_length=self.window_size,hidden_size=self.hidden_size,predict_length=self.predict_length)

        self.fc = nn.Linear(self.input_size,self.input_size)
        self.dropout = nn.Dropout(self.drop_out_rate)
        if self.mask:
            self.attn_mask = torch.triu(torch.ones(self.input_size,self.input_size),diagonal=1).to(self.device)
        else:
            self.attn_mask = None

        self.learning_rate = self.config["learning_rate"]













    def forward(self,x):

        batch_size, sequence_length, channels = x.size()

        attn_output, attn_weights = self.attention(x,self.attn_mask)

        attn_output = self.dropout(self.fc(attn_output))


        return attn_output, attn_weights



    def processData(self, data_train, data_test, shuffle=False):
        """
            对数据进行的预处理
            注意输出类型为可以直接送入训练的data_loader或张量

            :param data_train: 训练数据
            :param data_test: 测试数据

        """

        window_size = self.config["window_size"]
        batch_size = self.config["batch_size"]

        data_train = convertToWindow(data=data_train, window_size=window_size)
        data_test = convertToWindow(data=data_test, window_size=window_size)

        if shuffle:
            data_train = self.shuffle(data_train)

        train_dataset = TensorDataset(torch.tensor(data_train).float())
        test_dataset = TensorDataset(torch.tensor(data_test).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8)

        return (train_loader, test_loader)

    def fit(self, train_loader, write_log=False):

        self.train()


        base_optimizer = torch.optim.Adam

        rho = 0.5
        optimizer = SAM(self.parameters(), base_optimizer=base_optimizer, rho=rho,
                        lr=self.learning_rate, weight_decay=1e-5)





        l = nn.MSELoss(reduction='sum')

        epoch_loss = []


        for ep in range(self.epoch):
            ep = ep + 1
            l1s = []
            running_loss = 0
            for d in train_loader:
                optimizer.zero_grad()
                item = d[0].to(self.device)



                output,attn_weight = self.forward(item)

                loss = l(output[:,-1,:], item[:,-1,:])

                l1s.append(torch.mean(loss).item())

                running_loss += loss.item()

                loss.backward()

                optimizer.first_step(zero_grad=True)

                output, attn_weight = self.forward(item)
                loss = l(output[:, -1, :], item[:, -1, :])

                loss.backward()

                optimizer.second_step(zero_grad=True)



            # 计算当前epoch的平均损失
            epoch_loss.append(running_loss / len(train_loader))

            print(f'train epoch [{ep}/{self.epoch}],\t loss = {np.mean(l1s)}')

        identifier = self.config["identifier"]

        self.save()

        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": epoch_loss})

    def test(self, test_dataloader):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_dataloader: 测试数据

        """

        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.device)


                output, attn_weight = self.forward(item)

                loss = l(output[:, -1, :], item[:, -1, :])

                score.append(loss.sum(dim=-1).detach().cpu())


            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score

    def visualize(self,data_tensor):

        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')
        attn_weight = None
        with torch.no_grad():

                item = data_tensor.to(self.device)
                output, attn_weight = self.forward(item)


        print("attn hape:",attn_weight.shape)
        attn_weights_sample = attn_weight[-1]
        # 可视化注意力权重
        plt.figure(figsize=(12, 8))
        sns.heatmap(attn_weights_sample.detach().numpy(), cmap='viridis')
        plt.title("Attention Weights across Channels")
        plt.xlabel("Channels")
        plt.ylabel("Channels")
        plt.show()






