
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import countResult, findSegment
from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa


class LSTMAEV2(BaseModel):
    """
    《LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection》
    修改版，使用了时间窗口，仅重建最后一个时间点
    """
    def __init__(self,config):
        super(LSTMAEV2,self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]

        self.output_size = self.config["input_size"]

        self.encoder = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.decoder = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)


        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.device = self.config["device"]
        self.x_hat = None




    def forward(self,input):
        if self.x_hat == None:
            self.x_hat = torch.zeros_like(input)

        out, (hidden,cell) = self.encoder(input)

        out, (hidden,cell) = self.decoder(self.x_hat,(hidden,cell))
        x_hat = self.fc(hidden.permute((1,0,2))[:,-1,:])
        if len(x_hat.shape) < 3 :
            x_hat = x_hat.unsqueeze(dim=1)
        self.x_hat = x_hat.clone().detach()
        return x_hat



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

                item = d[0].to(self.device)



                y = self.forward(item)

                loss = F.mse_loss(y.squeeze(dim=1), item[:,-1,:], reduction='sum')



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
                item = d[0].to(self.device)

                y = self.forward(item)

                loss = F.mse_loss(y.squeeze(dim=1), item[:,-1,:], reduction='none')

                score.append(loss.sum(dim=-1).detach().cpu())

            score = torch.tensor(score).numpy()

            score = minMaxScaling(data = score,min_value= score.min(),max_value=score.max(),range_max=1,range_min=0)

        return score


