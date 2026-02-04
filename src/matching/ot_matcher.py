import ot
import numpy as np
from abc import ABC, abstractmethod

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"

class DataMatcher(ABC):

    @abstractmethod
    def match(self):
        pass


class OptimalTransportPolicy(DataMatcher):
    def __init__(self, X_labeled: np, X_unlabeled: np):

        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled
        self.reg = 0   ## should input the reg 
        self.ot_cost = ot.dist(self.X_labeled, self.X_unlabeled, p=2)
        self.probs_matrix = 0
        self.N = X_labeled.shape[0]
        self.M = X_unlabeled.shape[0]

    """Compute the prob_matrix of factual and counterfactual"""
    def match(self):
        if self.reg <= 0:
            self.probs_matrix = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost, numItermax=1000000
            )
        else:
            self.probs_matrix = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=1000000,
            )
        ot_matcher = self.convert_matrix_to_policy()
        return ot_matcher   

    def convert_matrix_to_policy(self):
        P = np.abs(self.probs_matrix) / np.abs(self.probs_matrix).sum()
        P += EPSILON
        P /= P.sum()
        return P