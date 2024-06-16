import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import lognorm, chi, norm
from torch.utils.data import TensorDataset, DataLoader

from Models.BaseModel import BaseModel
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import findSegment
from Utils.LogUtil import wirteLog
from Utils.ProtocolUtil import pa, apa


class FCAE(BaseModel):
    def __init__(self, config):
        super(FCAE).__init__()
        self.config: dict = config

        self.epoch: int = self.config["epoch"]
        self.input_size: int = self.config["input_size"]
        self.hidden_size: int = self.config["hidden_size"]
        self.learning_rate: int = self.config['learning_rate']
        self.window_size: int = self.config["window_size"]
        self.patience: int = self.config['patience']

        decoder_steps = 2 ** np.arange(max(np.ceil(np.log2(self.hidden_size)), 2), np.log2(self.input_length))
        decoder_steps = np.concatenate([[self.hidden_size], decoder_steps.repeat(2), [self.input_length]])
        encoder_steps = decoder_steps[::-1]

        layers = np.array([[nn.Linear(int(a), int(b), bias=True), nn.Tanh()] for a, b in encoder_steps.reshape(-1, 2)]).flatten()[:-1]
        self.encoder = nn.Sequential(*layers)

        layers = np.array([[nn.Linear(int(a), int(b), bias=True), nn.Tanh()] for a, b in decoder_steps.reshape(-1, 2)]).flatten()[:-1]
        self.decoder = nn.Sequential(*layers)

        self.device = self.config["device"]

    def forward(self, ts_batch, return_latent: bool = False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self.encoder(flattened_sequence.float())
        dec = self.decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence

    def processData(self, data_train, data_test, shuffle=False):
        window_size = self.config["window_size"]
        batch_size = self.config["batch_size"]

        data_train = convertToWindow(data=data_train, window_size=window_size)
        data_test = convertToWindow(data=data_test, window_size=window_size)
        if shuffle:
            data_train = self.shuffle(data_train)

        train_dataset = TensorDataset(torch.tensor(data_train).float())
        test_dataset = TensorDataset(torch.tensor(data_test).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        return train_loader, (train_loader, test_loader)

    def fit(self, train_loader, write_log=False):
        train_loss_by_epoch = []
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
        for ep in range(self.epoch):
            train_loss = []
            for (ts_batch) in train_loader:

                ts_batch = ts_batch.float().to(self.device)
                output = self.forward(ts_batch)
                loss = nn.MSELoss(reduction="mean")(output, ts_batch)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 乘以batch长度以纠正不完整batch的计算
                train_loss.append(loss.item() * len(ts_batch))
            scheduler.step()
            train_loss = np.mean(train_loss) / train_loader.batch_size
            train_loss_by_epoch.append(train_loss)
            print(f'train epoch [{ep}/{self.epoch}],\t loss = {train_loss}')

        identifier = self.config["identifier"]
        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": train_loss_by_epoch})

    @torch.no_grad()
    def test(self, test_dataloader, label):
        """
        test_dataloader包含训练集和测试集，train_loaders和test_loaders中每个通道对应一个dataloader
        """
        train_loader, test_loader = tuple(test_dataloader)
        self.eval()
        train_reconstr_scores = []
        test_reconstr_scores = []
        ts_batch = None
        # 测试
        for (ts_batch) in test_loader:
            ts_batch = ts_batch.float().to(self.device)
            output = self.forward(ts_batch)[:, -1]
            # 恢复为原始数据的格式
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
            test_reconstr_scores.append(error.cpu().detach().numpy())

        test_reconstr_scores = np.concatenate(test_reconstr_scores)

        for ts_batch in train_loader:
            ts_batch = ts_batch.float().to(self.device)
            output = self.forward(ts_batch)[:, -1]
            # 恢复为原始数据的格式
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
            train_reconstr_scores.append(error.cpu().detach().numpy())

        train_reconstr_scores = np.concatenate(train_reconstr_scores)

        # 填充第一个sequence前的空位
        multivar = (len(test_reconstr_scores.shape) > 1)
        if multivar:
            padding = np.zeros((len(ts_batch[0]) - 1, test_reconstr_scores.shape[-1]))
        else:
            padding = np.zeros(len(ts_batch[0]) - 1)

        error_tc_train = np.concatenate([padding, train_reconstr_scores]).reshape((-1))
        error_tc_test = np.concatenate([padding, test_reconstr_scores]).reshape((-1))

        # 计算异常分数
        distr_params = [self.__fit_univar_gaussian_distr(error_tc_train[:, i]) for i in range(error_tc_train.shape[1])]
        test_prob_scores = -1 * np.concatenate([self.__get_channel_probas(error_tc_test[:, i].reshape(-1, 1), distr_params[i]) for i in range(error_tc_test.shape[1])], axis=1)
        score_t_test = np.sum(test_prob_scores, axis=1)

        # 计算异常阈值
        test_anom_frac = (np.sum(label)) / len(label)
        threshold = np.nanpercentile(score_t_test, 100 * (1 - test_anom_frac), interpolation='higher')
        predict_label = np.where(score_t_test > threshold, 1, 0)

        anomaly_segments = findSegment(labels=label)
        pa_predict_label = pa(predict_label, anomaly_segments)
        apa_predict_label = apa(predict_label, anomaly_segments)

        pa_f1 = self.evaluate(predict_label=pa_predict_label, ground_truth_label=label, threshold=threshold, write_log=False)
        apa_f1 = self.evaluate(predict_label=apa_predict_label, ground_truth_label=label, threshold=threshold, write_log=False)
        return apa_f1, pa_f1

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
    def __get_channel_probas(scores_arr, params):
        constant_std = 0.000001
        assert ("mean" in params.keys() and ("std" in params.keys()) or "variance" in params.keys()), "参数缺失"
        if params["std"] == 0.0:
            params["std"] += constant_std
        distribution = norm(params["mean"], params["std"])
        probas = distribution.logsf(scores_arr)

        return probas


if __name__ == '__main__':
    pass
