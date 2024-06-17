import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from Models.BaseModel import BaseModel
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import findSegment
from Utils.LogUtil import wirteLog
from Utils.ProtocolUtil import pa, apa


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc


class ConvLSTM(nn.Module):
    def __init__(self, in_channels=32, h_channels=[32], kernel_size=3, seq_len=5, attention=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [in_channels] + h_channels
        self.hidden_channels = h_channels
        self.attention = attention
        self.kernel_size = kernel_size
        self.num_layers = len(h_channels)
        self.seq_len = seq_len
        self._all_layers = []
        self.flatten = nn.Flatten()
        self.alpha_i = None

        for i in range(self.num_layers):
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            self._all_layers.append(cell)

    def forward(self, input):
        """
        input with shape: (batch, num_channels, seq_len, height, width)
        """
        internal_state = []
        outputs = []
        for timestep in range(self.seq_len):
            x = input[:, :, timestep, ...]
            for i in range(self.num_layers):
                if timestep == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = self._all_layers[i].init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                (h, c) = internal_state[i]
                x, new_c = self._all_layers[i](x, h, c)
                internal_state[i] = (x, new_c)

            outputs.append(x)
        outputs = torch.stack(outputs).permute(1, 2, 0, 3, 4)

        if self.attention:
            alpha_i = [
                torch.einsum('ij,ik->i', self.flatten(outputs[:, :, i, ...]), self.flatten(outputs[:, :, -1, ...]))
                for i in range(self.seq_len)]
            alpha_i = torch.stack(alpha_i, dim=1) / self.seq_len
            alpha_i = F.softmax(alpha_i, dim=1)
            self.alpha_i = alpha_i
            x = torch.einsum('ijklm, ik -> ijlm', outputs, alpha_i)

        return outputs, (x, new_c)


class MSCRED(BaseModel):
    def __init__(self, config):
        super(MSCRED,self).__init__()

        self.config: dict = config

        self.epoch: int = self.config["epoch"]
        self.input_size: int = self.config["input_size"]
        self.hidden_size: int = self.config["hidden_size"]
        self.learning_rate: int = self.config['learning_rate']
        self.window_size: int = self.config["window_size"]
        self.patience: int = self.config['patience']
        self.step_max: int = self.config['step_max']
        self.win_size = [int((self.window_size / self.step_max) * i / 3) for i in range(1, 4)]

        num_timesteps = self.step_max
        attention = True
        self.Conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.ConvLSTM1 = ConvLSTM(in_channels=32, h_channels=[32], kernel_size=3, seq_len=num_timesteps, attention=attention)
        self.Conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.ConvLSTM2 = ConvLSTM(in_channels=64, h_channels=[64], kernel_size=3, seq_len=num_timesteps, attention=attention)
        self.Conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.ConvLSTM3 = ConvLSTM(in_channels=128, h_channels=[128], kernel_size=3, seq_len=num_timesteps, attention=attention)
        self.Conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.ConvLSTM4 = ConvLSTM(in_channels=256, h_channels=[256], kernel_size=3, seq_len=num_timesteps, attention=attention)
        self.Deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=1)
        self.Deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.Deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.device = self.config["device"]

        self.train_reconstr_scores = []

    def forward(self, x):
        """
        input X with shape: (batch, num_channels, seq_len, height, width)
        """
        x_c1_seq = F.selu(self.Conv1(x))
        _, (x_c1, _) = self.ConvLSTM1(x_c1_seq)

        x_c2_seq = F.selu(self.Conv2(x_c1_seq))
        _, (x_c2, _) = self.ConvLSTM2(x_c2_seq)

        x_c3_seq = F.selu(self.Conv3(x_c2_seq))
        _, (x_c3, _) = self.ConvLSTM3(x_c3_seq)

        x_c4_seq = F.selu(self.Conv4(x_c3_seq))
        _, (x_c4, _) = self.ConvLSTM4(x_c4_seq)

        x_d4 = F.selu(self.Deconv4.forward(x_c4, output_size=[x_c3.shape[-1], x_c3.shape[-2]]))

        x_d3 = torch.cat((x_d4, x_c3), dim=1)
        x_d3 = F.selu(self.Deconv3.forward(x_d3, output_size=[x_c2.shape[-1], x_c2.shape[-2]]))

        x_d2 = torch.cat((x_d3, x_c2), dim=1)
        x_d2 = F.selu(self.Deconv2.forward(x_d2, output_size=[x_c1.shape[-1], x_c1.shape[-2]]))

        x_d1 = torch.cat((x_d2, x_c1), dim=1)
        x_rec = F.selu(self.Deconv1.forward(x_d1, output_size=[x_c1.shape[-1], x_c1.shape[-2]]))

        return x_rec

    def processData(self, data, shuffle=False):
        window_size = self.config["window_size"]
        batch_size = self.config["batch_size"]

        data = convertToWindow(data=data, window_size=window_size)

        if shuffle:
            data = self.shuffle(data)

        data = self.create_signature_matrices(data)


        dataset = TensorDataset(torch.tensor(data).float())

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return data_loader

    def create_signature_matrices(self, sequences):
        n_dim = sequences.shape[2]
        matrices = np.zeros((sequences.shape[0], 3, self.step_max,
                             n_dim, n_dim))
        for i in range(sequences.shape[0]):
            raw_data_i = sequences[i]
            for k, w in enumerate(self.win_size):
                pad = self.window_size - self.step_max * w
                for j in range(self.step_max):
                    raw_data_ij = raw_data_i[(pad + j * w):(pad + (j + 1) * w)]
                    matrices[i, k, j] = np.dot(raw_data_ij.T, raw_data_ij) / w
        return matrices

    def fit(self, train_data, write_log=False):
        train_loader = self.processData(train_data)

        train_loss_by_epoch = []
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
        for ep in range(self.epoch):
            train_loss = []
            for ts_batch in train_loader:
                ts_batch = ts_batch[0]
                ts_batch = ts_batch.float().to(self.device)
                output = self.forward(ts_batch)

                loss = nn.MSELoss(reduction="mean")(output, ts_batch[:, :, -1, :, :])
                # 反向传播
                print("loss:",loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 乘以batch长度以纠正不完整batch的计算
                train_loss.append(loss.item() * len(ts_batch))

                # 恢复为原始数据的格式
                error = nn.L1Loss(reduction='none')(output, ts_batch[:, :, -1, :, :])
                self.train_reconstr_scores.append(error.cpu().detach().numpy())



            scheduler.step()
            train_loss = np.mean(train_loss) / train_loader.batch_size
            train_loss_by_epoch.append(train_loss)
            print(f'train epoch [{ep}/{self.epoch}],\t loss = {train_loss}')



        identifier = self.config["identifier"]
        if write_log:
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "train_loss", {"epoch_loss": train_loss_by_epoch})

    @torch.no_grad()
    def test(self, test_data):
        """

        """
        test_loader = self.processData(test_data)

        self.eval()

        test_reconstr_scores = []
        ts_batch = None
        # 测试
        for ts_batch in test_loader:
            ts_batch = ts_batch[0]
            ts_batch = ts_batch.float().to(self.device)
            output = self.forward(ts_batch)
            # 恢复为原始数据的格式
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, :, -1, :, :])
            test_reconstr_scores.append(error.cpu().detach().numpy())

        test_reconstr_scores = np.concatenate(test_reconstr_scores)


        train_reconstr_scores = np.concatenate(self.train_reconstr_scores)

        # # 填充第一个sequence前的空位
        # multivar = (len(test_reconstr_scores.shape) > 1)
        # if multivar:
        #     padding = np.zeros((len(ts_batch[0]) - 1, test_reconstr_scores.shape[-1]))
        # else:
        #     padding = np.zeros(len(ts_batch[0]) - 1)
        #
        # error_tc_train = np.concatenate([padding, train_reconstr_scores]).reshape((-1))
        # error_tc_test = np.concatenate([padding, test_reconstr_scores]).reshape((-1))

        # 计算异常分数
        distr_params = [self.__fit_univar_gaussian_distr(train_reconstr_scores[:, i]) for i in range(train_reconstr_scores.shape[1])]
        test_prob_scores = -1 * np.concatenate([self.__get_channel_probas(test_reconstr_scores[:, i].reshape(-1, 1), distr_params[i]) for i in range(test_reconstr_scores.shape[1])], axis=1)
        score = np.sum(test_prob_scores, axis=1)

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score


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


    def setThreshold(self,**kwargs):
        # 计算异常阈值
        label = kwargs["label"]
        score = kwargs["score"]
        test_anom_frac = (np.sum(label)) / len(label)
        self.threshold = np.nanpercentile(score, 100 * (1 - test_anom_frac), interpolation='higher')



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
