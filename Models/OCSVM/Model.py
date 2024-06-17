import numpy as np
import torch
from scipy.stats import norm
from sklearn import svm

from Models.BaseModel import BaseModel
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow
from Utils.EvalUtil import findSegment
from Utils.ProtocolUtil import pa, apa


class OCSVM(BaseModel):
    def __init__(self, config):
        super(OCSVM,self).__init__()
        self.config: dict = config

        self.kernel: str = self.config["kernel"]
        self.gamma: str = self.config["gamma"]
        self.degree: int = self.config["degree"]
        self.coef0: float = self.config['coef0']
        self.window_size: int = self.config["window_size"]
        self.tol: float = self.config['tol']
        self.cache_size: int = self.config["cache_size"]
        self.shrinking: bool = self.config['shrinking']
        self.nu: float = self.config['nu']

        self.model = None



    def fit(self, train_data, write_log=False):

        model = svm.OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, tol=self.tol, nu=self.nu, shrinking=self.shrinking,
                            cache_size=self.cache_size, max_iter=-1, verbose=False)
        model.fit(train_data)
        self.model = model

    @torch.no_grad()
    def test(self, test_data):

        predict = self.model.predict(test_data)


        score = -predict
        score =  minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score


if __name__ == '__main__':
    pass
