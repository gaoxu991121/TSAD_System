
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


class LSTMEncoder(BaseModel):
    def __init__(self,input_size,hidden_size,latent_size):
        super(LSTMEncoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.z_mean_layer = nn.Linear(hidden_size,latent_size)
        self.z_log_var_layer = nn.Linear(hidden_size,latent_size)


    def forward(self,x):
        #hidden/cell  shape:[num_layers,batch_size,hidden_size]

        # input shape: [batch_size,window_size,features]
        lstm_output,(hidden,cell) = self.lstm(x)

        hidden = hidden.permute(1,0,2)

        z_mean = self.z_mean_layer(hidden)
        z_log_var = self.z_log_var_layer(hidden)
        epsilon = torch.randn_like(z_log_var)
        z = z_mean + z_log_var.exp() * epsilon

        #z shape: [batch,1 , latent_size]
        return z,z_mean,z_log_var

class LSTMDecoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_size):
        super(LSTMDecoder,self).__init__()
        self.lstm = nn.LSTM(input_size=latent_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.x_mean_layer = nn.Linear(hidden_size, input_size)
        self.x_log_var_layer = nn.Linear(hidden_size, input_size)

    def forward(self,z):
        #cell shape: torch.Size([1, 128, 28])
        #hidden shape: torch.Size([1, 128, 28])
        output1,(hidden,cell) = self.lstm(z)

        hidden = hidden.permute(1, 0, 2)
        x_mean = self.x_mean_layer(hidden)
        x_log_var = self.x_log_var_layer(hidden)
        return x_mean,x_log_var

class LSTMVAE(BaseModel):
    def __init__(self,config):
        '''
        《A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder》 实现
        '''
        super(LSTMVAE,self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.latent_size = self.config["latent_size"]

        self.encoder = LSTMEncoder(self.input_size,self.hidden_size,self.latent_size)
        self.decoder = LSTMDecoder(self.input_size,self.hidden_size,self.latent_size)
        self.device = self.config["device"]

    def forward(self,input):

        z,z_mean,z_log_var = self.encoder(input)
        x_mean,x_std = self.decoder(z)

        return (z_mean,z_log_var),(x_mean,x_std)

    def processData(self,data_train,data_test,shuffle = False):
        """
            对数据进行的预处理
            注意输出类型为可以直接送入训练的data_loader或张量

            :param data_train: 训练数据
            :param data_test: 测试数据

        """


        window_size = self.config["window_size"]
        batch_size = self.config["batch_size"]

        data_train = convertToWindow(data = data_train, window_size = window_size)
        data_test = convertToWindow(data = data_test, window_size = window_size)

        if shuffle:
            data_train = self.shuffle(data_train)


        train_dataset = TensorDataset(torch.tensor(data_train).float())
        test_dataset = TensorDataset(torch.tensor(data_test).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return (train_loader,test_loader)


    def fit(self,train_loader,write_log = False):

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

                (z_mean, z_log_var), (x_mean, x_std) = self.forward(item)

                reconstruction_loss = F.mse_loss(x_mean.squeeze(), item[:, -1, :].squeeze(), reduction='sum')

                # KL散度
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = reconstruction_loss + kl_loss

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




    def test(self,test_dataloader):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_dataloader: 测试数据

        """


        self.eval()
        score = []
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.device)

                (z_mean, z_log_var), (x_mean, x_std) = self.forward(item)
                loss = F.mse_loss(x_mean.squeeze(), item[:, -1, :].squeeze(), reduction="none")

                if item.shape[-1] > 1:
                    loss = loss.sum(dim=-1)

                score.append(loss.detach().cpu())

            score = torch.concatenate(score,dim=0).numpy()

            score = minMaxScaling(data = score,min_value= score.min(),max_value=score.max(),range_max=1,range_min=0)

        return score

