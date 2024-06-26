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


class UAEModel(nn.Module):
    def __init__(self, hidden_size, sequence_length, device, n_features=1, bias=True):
        super(UAEModel, self).__init__()
        input_length = n_features * sequence_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b), bias=bias), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self.encoder = nn.Sequential(*layers)

        layers = np.array([[nn.Linear(int(a), int(b), bias=bias), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self.decoder = nn.Sequential(*layers)

        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, ts_batch, return_latent: bool = False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self.encoder(flattened_sequence.float())
        dec = self.decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence


class UAE(BaseModel):
    def __init__(self, config):
        super(UAE,self).__init__()
        self.config: dict = config

        self.epoch: int = self.config["epoch"]
        self.input_size: int = self.config["input_size"]
        self.hidden_size: int = self.config["hidden_size"]
        self.learning_rate: int = self.config['learning_rate']
        self.window_size: int = self.config["window_size"]
        self.patience: int = self.config['patience']
        self.device = self.config["device"]

        self.error_tc_train = None

        self.model = []


    def fit(self, train_data, write_log=False):

        train_loader = self.processData(train_data)
        train_intermediate_scores = []
        for channal_id in range(self.input_size):

            model = UAEModel(hidden_size=self.hidden_size, sequence_length=self.window_size, device=self.device)
            model.to(self.device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            train_loss_by_epoch = []

            train_reconstr_scores = []


            for ep in range(self.epoch):
                train_loss = []

                for ts_batch in train_loader:
                    ts_batch = ts_batch[0][:,:,channal_id]
                    ts_batch = ts_batch.float().to(model.device)
                    output = model(ts_batch)

                    loss = nn.MSELoss(reduction="mean")(output, ts_batch)

                    # 反向传播
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 乘以batch长度以纠正不完整batch的计算
                    train_loss.append(loss.item() * len(ts_batch))

                    error = nn.L1Loss(reduction='none')(output[:, -1], ts_batch[:, -1])
                    if ep == self.epoch - 1:
                        train_reconstr_scores.append(error.cpu().detach().numpy())

                if ep == self.epoch - 1:
                    train_reconstr_scores = np.concatenate(train_reconstr_scores)

                train_loss = np.mean(train_loss) / train_loader.batch_size
                train_loss_by_epoch.append(train_loss)
                print(f'train epoch [{ep}/{self.epoch}],\t loss = {train_loss}')

            train_intermediate_scores.append(train_reconstr_scores)
            self.error_tc_train = np.array(train_intermediate_scores).T

            self.model.append(model)

            identifier = self.config["identifier"]
            if write_log:
                wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": train_loss_by_epoch})

    @torch.no_grad()
    def test(self, test_data):

        test_loader = self.processData(test_data)

        loss_funcation = nn.MSELoss(reduction='none')
        test_intermediate_scores = []

        for channal_id in range(self.input_size):
            model = self.model[channal_id]
            model.eval()
            test_reconstr_scores = []

            for ts_batch in test_loader:
                ts_batch = ts_batch[0][:, :, channal_id]
                ts_batch = ts_batch.float().to(model.device)
                output = model(ts_batch)
                # 恢复为原始数据的格式
                # error = nn.L1Loss(reduction='none')(output[:, -1], ts_batch[:, -1])
                error = loss_funcation(output[:, -1],ts_batch[:, -1])
                test_reconstr_scores.append(error.detach().cpu().numpy())


            test_reconstr_scores = np.concatenate(test_reconstr_scores)
            test_intermediate_scores.append(test_reconstr_scores)


        error_tc_test = np.array(test_intermediate_scores).T



        # 计算异常分数
        # distr_params = [self.__fit_univar_gaussian_distr(self.error_tc_train[:, i]) for i in range(self.error_tc_train.shape[1])]
        # test_prob_scores = -1 * np.concatenate([self.__get_channel_probas(error_tc_test[:, i].reshape(-1, 1), distr_params[i], logcdf=True) for i in range(error_tc_test.shape[1])],
        #                                        axis=1)
        score = np.sum(error_tc_test, axis=1)

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
