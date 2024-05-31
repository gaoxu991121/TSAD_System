
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import countResult, findSegment
from Utils.LogUtil import wirteLog
from torch.nn import functional as F

from Utils.ProtocolUtil import pa


class LSTM(nn.Module):
    """
    《Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding》

    """
    def __init__(self,config):
        super(LSTM,self).__init__()
        self.config = config

        self.epoch = self.config["epoch"]
        self.input_size = 1
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.drop_out_rate = self.config["drop_out_rate"]
        self.output_size = self.config["input_size"]
        self.window_size = self.config["window_size"]

        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)

        self.dropout = nn.Dropout(self.drop_out_rate)
        self.fc = nn.Linear(self.hidden_size, 1)  # 1 是输出维度




    def forward(self,input):
        '''
        本处的实现是多个维度共享参数。通过每个batch展开为1维来实现。

        '''
        shape = input.shape
        # input = input.reshape(-1, self.window_size-1, 1)
        # #hidden shape: torch.Size([num_layer, batch_size*features, hidden_size])
        # #out shape:[batch_size*features, sequence_length, hidden_dim]
        # out,(hidden,cell) = self.lstm(input)
        #
        # hidden = hidden[-1,:,:].unsqueeze(dim = 1)
        #
        # out = self.dropout(hidden)
        #
        # #out shape:[batch_size*features, hidden_size]
        #
        # out = self.fc(out).reshape([shape[0],shape[-1]])  #取 LSTM 输出的最后一个时间步作为预测结果

        data_list = []

        for i in range(shape[-1]):
            data = input[:,:,i].unsqueeze(dim=-1)
            out, (hidden, cell) = self.lstm(data)
            out = self.dropout(hidden).permute((1,0,2))[:,-1,:]
            out = self.fc(out)

            data_list.append(out.squeeze())

        out = torch.stack(data_list,dim=1)

        return out

    def processData(self,data_train,data_test):
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

        for ep in range(self.epoch):
            l1s = []
            for d in train_loader:
                optimizer.zero_grad()
                item = d[0]


                y = self.forward(item[:,:-1,:])

                loss = F.mse_loss(y, item[:,-1,:], reduction='sum')
                print("loss:",loss)

                l1s.append(torch.mean(loss).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()  # 在每个epoch后更新学习率

            tqdm.write(f'train epoch [{ep}/{self.epoch}],\t loss = {np.mean(l1s)}')


        identifier = self.config["identifier"]
        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier,"train_loss",{"train_loss":l1s})


    def getThreshold(self):
        threshold = 0.5
        if self.config["threshold"] != None:
            threshold = self.config["threshold"]

        return threshold


    def test(self,test_dataloader):
        """
             在测试集上进行测试，输出的是归一到[0,1]的numpy数组类型的异常得分
             :param test_dataloader: 测试数据

        """


        self.eval()
        score = []
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0]


                y = self.forward(item[:, :, :-1])

                loss = F.mse_loss(y, item[:,-1,:], reduction='sum')


                score.append(loss.detach())

            score = torch.concatenate(score,dim=0).numpy()

            score = minMaxScaling(data = score,min_value= score.min(),max_value=score.max(),range_max=1,range_min=0)

        return score

    def predict(self,anomaly_score,threshold,ground_truth_label = [],protocol = ""):
        """
            根据异常得分以及阈值输出预测结果，在此函数内调用评估协议或其他处理
            :param anomaly_score: 异常得分
            :param threshold: 阈值
            :param ground_truth_label: 真值标签，不使用则不需要传
            :param protocol: 调用的评估协议，不使用则不需要传

       """

        predict_label = np.where(anomaly_score > threshold, 1, 0)

        if protocol == "pa":
            anomaly_segments = findSegment(labels=ground_truth_label)
            predict_label = pa(predict_label, anomaly_segments)



        return predict_label


    def evaluate(self,predict_label,ground_truth_label):
        """
            根据预测标签以及真值标签，给出评估结果。此处给出了f1，可添加其他
            :param predict_label: 预测标签
            :param ground_truth_label: 真值标签，不使用则不需要传
        """


        (tp, fp, tn, fn) = countResult(predict_labels=predict_label, ground_truth=ground_truth_label)

        if (tp + fn + fp) == 0:
            f1 = 0
        else:
            f1 = (2 * tp) / (2 * (tp + fn + fp))

        return f1

