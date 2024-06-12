import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.PE import PE
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import countResult, findSegment
from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa


class Encoder(nn.Module):
    """
    《 TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data 》

    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = self.config["input_size"]

        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]
        self.num_heads = self.config["num_heads"]

        self.pe = PE(self.input_size, max_len=self.window_size*2)

        self.encoder_attn = nn.MultiheadAttention(embed_dim=self.input_size * 2, num_heads=self.num_heads,
                                                  dropout=self.drop_out_rate, batch_first=True)
        self.encoder_layernorm1 = nn.LayerNorm(self.input_size * 2)
        self.encoder_layernorm2 = nn.LayerNorm(self.input_size)

        self.encoder_dropout1 = nn.Dropout(p=self.drop_out_rate)
        self.encoder_fc1 = nn.Linear(self.input_size * 2, self.input_size * 2)
        self.encoder_fc2 = nn.Linear(self.input_size * 2, self.input_size )

        self.win_encoder_attn1 = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.num_heads,
                                                      dropout=self.drop_out_rate, batch_first=True)

        self.win_encoder_attn2 = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.num_heads,
                                                      dropout=self.drop_out_rate, batch_first=True)


        self.win_encoder_layernorm1 = nn.LayerNorm(self.input_size)
        self.win_encoder_layernorm2 = nn.LayerNorm(self.input_size)

        self.fc2 = nn.Linear(self.input_size, self.input_size)

        self.attn_mask = torch.triu(torch.ones(self.window_size,self.window_size),diagonal=1).to(dtype=torch.float)

        self.divice = self.config["device"]


    def forward(self, c, w, focus_score):

        focus_score = self.pe(focus_score)
        c = self.pe(c)
        c_focus_score = torch.concatenate((c, focus_score), dim=-1)
        attn_output = c_focus_score +   self.encoder_attn(c_focus_score,c_focus_score,c_focus_score)[0]
        attn_output = self.encoder_layernorm1(attn_output)

        attn_output = attn_output + self.encoder_dropout1(self.encoder_fc1(attn_output))
        attn_output = self.encoder_fc2(attn_output)
        encoder_output = self.encoder_layernorm2(attn_output)


        w = self.pe(w)
        window_attn_output =  w + self.win_encoder_attn1(w,w,w,attn_mask = self.attn_mask)[0]
        window_attn_output = self.win_encoder_layernorm1(window_attn_output)


        window_attn_output = window_attn_output + self.win_encoder_attn2(encoder_output,encoder_output,window_attn_output,attn_mask = self.attn_mask)[0]
        window_attn_output = self.win_encoder_layernorm2(window_attn_output)
        return window_attn_output


class Decoder(nn.Module):
    """
    《 TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data 》

    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.input_size = self.config["input_size"]

        self.drop_out_rate = self.config["drop_out_rate"]
        self.output_size = self.config["input_size"]
        self.window_size = self.config["window_size"]
        self.num_heads = self.config["num_heads"]


        self.fc = nn.Linear(self.input_size,self.output_size)
        self.dropout = nn.Dropout(p=self.drop_out_rate)
        self.sigmod = nn.Sigmoid()

        self.attention = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self,data):
        weights = self.attention(data)  # [batch, sequence, 1]
        data = torch.sum(weights * data, dim=1, keepdim=True)

        data = self.fc(data)
        data = self.dropout(data)
        data = self.sigmod(data)

        return data












class TRANAD(BaseModel):
    """
    《 TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data 》

    """

    def __init__(self, config):
        super(TRANAD, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]


        self.divice = self.config["device"]

        self.encoder = Encoder(self.config)
        self.decoder1 = Decoder(self.config)
        self.decoder2 = Decoder(self.config)







    def forward(self,  w):

        focus_score = torch.zeros_like(w)

        #phase 1
        x1 = self.decoder1(self.encoder(w, w, focus_score))
        focus_score = (x1 - w)**2


        #phase 2
        x2 = self.decoder2(self.encoder(w, w, focus_score))


        return x1,x2



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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return (train_loader, test_loader)

    def fit(self, train_loader, write_log=False):

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



                x1,x2 = self.forward(item)


                loss = (1 / ep) * l(x1[:, -1, :], item[:, -1, :]) + (1 - 1 / ep) * l(x2[:, -1, :], item[:, -1, :])



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
                item = d[0].to(self.divice)

                x1, x2 = self.forward(item)

                loss = 0.5*l(x2[:, -1, :], item[:, -1, :]) + 0.5*l(x1[:, -1, :], item[:, -1, :])

                score.append(loss.sum(dim=-1).detach().cpu())


            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score

