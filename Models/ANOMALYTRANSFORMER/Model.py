import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models.BaseModel import BaseModel
from Models.Layers.AnomalyMultiHeadAttention import AnomalyBlock, AnomalyBlockList
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow

from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa
import matplotlib.pyplot as plt
import seaborn as sns



class ANOMALYTRANSFORMER(BaseModel):

    def __init__(self, config):
        super(ANOMALYTRANSFORMER, self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.drop_out_rate = self.config["drop_out_rate"]

        self.window_size = self.config["window_size"]


        self.divice = self.config["device"]

        self.num_heads = self.config["num_heads"]
        self.num_layers = self.config["num_layers"]

        self.k = self.config["lambda"]

        self.blocks = AnomalyBlockList(
            [
                AnomalyBlock(self.input_size,self.num_heads,self.window_size,self.drop_out_rate)

            for i in range(self.num_layers) ]
        )

        self.pe = PE(self.input_size, max_len=self.window_size)




    def forward(self,  x):

        x = self.pe(x)

        output, series, prior = self.blocks(x)

        return output, series, prior




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

    def getKlLoss(self,p, q):


        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.sum(res, dim=-1)

    def getAssDis(self, prior, series):

        return self.getKlLoss(prior,series) + self.getKlLoss(prior,series)

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



                output_list, series_list, prior_list = self.forward(item)


                num_layers = len(prior_list)

                series_kl_loss = 0
                prior_kl_loss = 0



                for i in range(num_layers):
                    prior = prior_list[i]
                    serie = series_list[i][:,-1,:,:]

                    series_kl_loss +=  self.getAssDis(prior = prior,series=serie.detach()).mean()
                    prior_kl_loss += self.getAssDis(prior = prior.detach(),series=serie).mean()


                series_kl_loss = (series_kl_loss / num_layers).sum(dim=-1)
                prior_kl_loss = (prior_kl_loss / num_layers).sum(dim=-1)



                reco_loss = F.mse_loss(output_list[-1],item,reduction='sum')
                print("recon loss:",reco_loss)
                print("prior_kl_loss:", prior_kl_loss)
                print("series_kl_loss:", series_kl_loss)
                loss1 = reco_loss - self.k * series_kl_loss

                loss2 = reco_loss + self.k * prior_kl_loss


                l1s.append(torch.mean(loss2).item())

                running_loss += loss2.item()

                loss1.backward(retain_graph=True)
                loss2.backward()

                optimizer.zero_grad()

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

        l = nn.MSELoss(reduction='sum')

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(self.divice)

                output_list, series_list, prior_list = self.forward(item)

                num_layers = len(prior_list)

                ass_loss = 0

                for i in range(num_layers):
                    ass_loss += self.getAssDis(prior_list[i], series_list[i])

                ass_loss = ass_loss / num_layers
                anomaly_score = torch.softmax(-ass_loss,dim=-1) * l(output_list[-1],item)







                score.append(anomaly_score.mean().detach().cpu())


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






