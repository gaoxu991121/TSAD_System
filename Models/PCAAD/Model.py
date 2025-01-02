import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Models.BaseModel import BaseModel
from Preprocess.Normalization import instanceNormalization, minMaxScaling
from Utils.EvalUtil import findSegment
from Utils.ProtocolUtil import pa, apa


class PCAAD(BaseModel):
    def __init__(self, config):
        super(PCAAD,self).__init__()
        self.config: dict = config
        self.explained_var: float = self.config['explained_var']
        self.scaler = None
        self.model = None

    def processData(self, data, shuffle=False):
        if len(data.shape) >= 3:
            data = data[:,-1,:].squeeze()
        data = instanceNormalization(data,data.shape[-1])
        return data

    def fit(self, train_data, write_log=False):
        train_loader = self.processData(train_data)
        self.model = PCA(n_components=self.explained_var)
        print("Fitting PCAAD")
        self.model.fit(train_loader)
        print("Done fitting PCAAD. n_components for explained variance {} = {}".format(self.explained_var, self.model.n_components_))

    def test(self, test_data):
        test_dataloader = self.processData(test_data)

        recons_tc = np.dot(self.model.transform(test_dataloader), self.model.components_)
        score = (test_dataloader - recons_tc) ** 2
        score = np.sum(score, axis=1)

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score




