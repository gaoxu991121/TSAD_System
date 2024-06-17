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



class OmniAnomaly(BaseModel):


    def __init__(self, config):
        super(OmniAnomaly, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.latent_size = self.config["latent_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]
        self.beta = 0.01

        self.device = self.config["device"]

        self.lstm = nn.GRU(self.input_size, self.hidden_size, 2, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.PReLU(),
            nn.Linear(self.hidden_size, 2 * self.latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size), nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.PReLU(),
            nn.Linear(self.hidden_size, self.input_size), nn.Sigmoid(),
        )

        self.learning_rate = self.config["learning_rate"]

    def forward(self, x, hidden=None):

        hidden = torch.rand(2, x.shape[0], self.hidden_size, dtype=torch.float64).float().to(self.device) if hidden is not None else hidden
        out, hidden = self.lstm(x, hidden)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.latent_size, self.latent_size], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std

        ## Decoder
        x = self.decoder(x)

        return x, mu, logvar, hidden



    def fit(self, train_data, write_log=False):
        train_loader = self.processData(train_data)
        self.train()


        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        l = nn.MSELoss(reduction='none')

        epoch_loss = []


        for ep in range(self.epoch):
            ep = ep + 1
            l1s = []
            running_loss = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                item = batch[0].to(self.device)

                y_pred, mu, logvar, hidden = self.forward(item, hidden if i else None)

                MSE = l(y_pred, item).sum()
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = MSE + self.beta * KLD

                running_loss += loss.item()
                l1s.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



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

        test_loader = self.processData(test_data)

        self.eval()
        score = []

        l = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                item = batch[0]
                y_pred, mu, logvar, hidden = self.forward(item, hidden if i else None)
                # 将张量拆分成 128 个形状为 (5,) 的张量

                score.append(l(y_pred[:, -1, :], item[:, -1, :]).sum(dim=-1))

            score = torch.cat([tensor for tensor in score], dim=0).squeeze()

            score = score.cpu().detach().numpy()
            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score






