import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from Models.BaseModel import BaseModel
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import findSegment
from Utils.LogUtil import wirteLog
from scipy.stats import lognorm, chi, norm

from Utils.ProtocolUtil import pa, apa

class LSTMBase(nn.Module):

    def __init__(self,config):
        super(LSTMBase, self).__init__()
        self.config = config
        self.input_size = 1
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.drop_out_rate = self.config["drop_out_rate"]
        self.output_size = self.config["input_size"]
        self.window_size = self.config["window_size"]
        self.device = self.config["device"]
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.fc = nn.Linear(self.hidden_size, 1)  # 1 是输出维度

    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.dropout(self.fc(x))
        return x


class NASALSTM(BaseModel):
    def __init__(self, config):
        super(NASALSTM,self).__init__()
        self.config: dict = config


        self.epoch = self.config["epoch"]
        self.input_size = 1
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.drop_out_rate = self.config["drop_out_rate"]
        self.output_size = self.config["input_size"]
        self.window_size = self.config["window_size"]
        self.learning_rate = self.config["learning_rate"]

        self.dropout = nn.Dropout(self.drop_out_rate)


        self.device = self.config["device"]
        self.model = []


    def fit(self, train_data, write_log=False):

        train_loader = self.processData(train_data)

        for channal_id in range(self.input_size):

            model = LSTMBase(config=self.config)
            model.to(self.device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            train_loss_by_epoch = []


            for ep in range(self.epoch):
                train_loss = []

                for ts_batch in train_loader:
                    ts_batch = ts_batch[0][:,:,channal_id]
                    ts_batch = ts_batch.float().unsqueeze(dim=-1).to(self.device)
                    print("ts_batch shape:",ts_batch.shape)
                    output = model(ts_batch)

                    print("output shape:",output.shape)
                    loss = nn.MSELoss(reduction="mean")(output, ts_batch)

                    # 反向传播
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()


                train_loss = np.mean(train_loss) / train_loader.batch_size
                train_loss_by_epoch.append(train_loss)
                print(f'train epoch [{ep}/{self.epoch}],\t loss = {train_loss}')


            self.model.append(model)

            identifier = self.config["identifier"]
            if write_log:
                wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": train_loss_by_epoch})

    @torch.no_grad()
    def test(self, test_data):

        test_loader = self.processData(test_data)


        score = []
        for channal_id in range(self.input_size):
            model = self.model[channal_id]
            model.eval()
            test_reconstr_scores = []

            for ts_batch in test_loader:
                ts_batch = ts_batch[0][:, :, channal_id]
                ts_batch = ts_batch.float().unsqueeze(dim=-1).to(self.device)
                output = model(ts_batch)
                # 恢复为原始数据的格式
                error = nn.L1Loss(reduction='none')(output[:, -1], ts_batch[:, -1])
                test_reconstr_scores.append(error.cpu().detach().numpy())


            score.append(test_reconstr_scores)

        score = np.array(score)
        print("score shape:",score.shape)

        score = np.sum(score, axis=1)
        print("score 2 shape:",score.shape)
        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)
        return score


    def setThreshold(self,**kwargs):
        # 计算异常阈值
        label = kwargs["label"]
        score = kwargs["score"]
        test_anom_frac = (np.sum(label)) / len(label)
        self.threshold = np.nanpercentile(score, 100 * (1 - test_anom_frac), interpolation='higher')



    def predictEvaluate(self, test_data, label, protocol=""):
        anomaly_scores = self.test(test_data)
        self.setThreshold(score = anomaly_scores,label = label)
        # predict anomaly based on the threshold
        threshold = self.getThreshold()
        predict_labels = self.decide(anomaly_score=anomaly_scores, threshold=threshold, ground_truth_label=label,
                                     protocol=protocol)

        # evaluate
        f1 = self.evaluate(predict_label=predict_labels, ground_truth_label=label, threshold=threshold, write_log=False)
        return f1


    @staticmethod
    def __fit_univar_gaussian_distr(scores_arr: np.ndarray):
        distributed_params = {'distr': 'univar_gaussian'}
        constant_std = 0.000001
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        if std == 0.0:
            std += constant_std
        distributed_params["mean"] = mean
        distributed_params["std"] = std

        return distributed_params

    @staticmethod
    def __get_channel_probas(scores_arr, params, logcdf=False):
        constant_std = 0.000001
        assert ("mean" in params.keys() and ("std" in params.keys()) or "variance" in params.keys()), "参数缺失"
        if "std" in params.keys():
            if params["std"] == 0.0:
                params["std"] += constant_std
            distribution = norm(params["mean"], params["std"])
        else:
            distribution = norm(params["mean"], np.sqrt(params["variance"]))

        if logcdf:
            probas = distribution.logsf(scores_arr)
        else:
            probas = distribution.logpdf(scores_arr)

        return probas


if __name__ == '__main__':
    pass
