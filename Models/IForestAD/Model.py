import numpy as np

from Models.BaseModel import BaseModel
import random

from pyod.models.iforest import IForest

from Preprocess.Normalization import minMaxScaling
from Utils.EvalUtil import findSegment
from Utils.ProtocolUtil import pa, apa

class IForestAD(BaseModel):
    def __init__(self, config):
        super(IForestAD,self).__init__()
        self.config: dict = config

        self.n_trees = config['n_trees']  # The number of decision trees (base estimators) in the forest (ensemble).
        self.max_samples = config['max_samples']  # The number of samples to draw from X to train each base estimator: `max_samples * X.shape[0]`.
        # If unspecified (`null`), then `max_samples=min(256, n_samples)`."
        self.max_features = config['max_features']  # The number of features to draw from X to train each base estimator: `max_features * X.shape[1]`.
        self.bootstrap = config['bootstrap']  # If True, individual trees are fit on random subsets of the training data sampled with replacement.
        # If False, sampling without replacement is performed.
        self.random_state = config['random_state']  # Seed for random number generation.
        self.verbose = config['verbose']  # Controls the verbosity of the tree building process logs.
        self.n_jobs = config['n_jobs']  # The number of jobs to run in parallel. If -1, then the number of jobs is set to the number of cores.
        self.contamination = config["contamination"]

    def fit(self, train_data, write_log=False):
        pass

    def test(self, test_data):


        clf = IForest(
            contamination=self.contamination,
            n_estimators=self.n_trees,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )

        clf.fit(test_data)
        score = clf.decision_scores_
        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        return score
