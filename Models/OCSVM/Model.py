import numpy as np
import torch
from scipy.stats import norm
from thundersvm import OneClassSVM

from Models.BaseModel import BaseModel
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import findSegment
from Utils.ProtocolUtil import pa, apa


class UAE(BaseModel):
    def __init__(self, config):
        super(UAE).__init__()
        self.config: dict = config

        self.kernel: str = self.config["kernel"]
        self.gamma: str = self.config["gamma"]
        self.degree: int = self.config["degree"]
        self.coef0: float = self.config['coef0']
        self.window_size: int = self.config["window_size"]
        self.tol: float = self.config['tol']
        self.cache_size: int = self.config["cache_size"]
        self.shrinking: bool = self.config['nu']
        self.nu: float = self.config['nu']

        self.model = None

    def processData(self, data_train, data_test, shuffle=False):
        window_size = self.config["window_size"]

        data_train = convertToWindow(data=data_train, window_size=window_size)
        data_test = convertToWindow(data=data_test, window_size=window_size)
        if shuffle:
            data_train = self.shuffle(data_train)
        return data_train, data_test

    def fit(self, train_loader, write_log=False):
        train_sequences = train_loader.reshape(train_loader.shape[0], -1)
        model = OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, tol=self.tol, nu=self.nu, shrinking=self.shrinking,
                            cache_size=self.cache_size, max_iter=-1, verbose=False)
        model.fit(train_sequences)
        self.model = model

    @torch.no_grad()
    def test(self, test_dataloader, label):
        padding = np.zeros(self.sequence_length - 1)

        sequences = test_dataloader.reshape(test_dataloader.shape[0], -1)
        score_t = self.__convert_pred(self.model.predict(sequences).reshape(-1))
        score_t_test = np.concatenate([padding, score_t])

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


if __name__ == '__main__':
    pass
