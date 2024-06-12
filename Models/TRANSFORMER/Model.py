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





class TRANSFORMER(BaseModel):
    """

    """

    def __init__(self, config):
        super(TRANSFORMER, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]


        self.divice = self.config["device"]

        self.num_heads = self.config["num_heads"]


        self.transformer = nn.Transformer(self.input_size,nhead=self.num_heads,batch_first=True,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=256)

        self.fc = nn.Linear(self.input_size,self.input_size)






    def forward(self,  x,target):

        x = self.transformer(x,target)

        x = self.fc(x)
        return x



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



                output = self.forward(item,item[:,-1,:].unsqueeze(dim=1))

                loss = l(output, item[:,-1,:].unsqueeze(dim=1))



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

                output = self.forward(item, item[:, -1, :].unsqueeze(dim=1))
                loss = l(output[:, -1, :], item[:, -1, :])
                score.append(loss.sum(dim=-1).detach().cpu())


            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score








