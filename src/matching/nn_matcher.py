from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
EPSILON = 1e-20

class DataMatcher(ABC):

    @abstractmethod
    def match(self):
        pass


class NearestNeighborDataMatcher(DataMatcher):

    def __init__(self, X_labeled: np.ndarray, X_unlabeled: np.ndarray):
        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled

    def match(self, n_neighbors: Optional[int] = 1) -> np.ndarray:
        # Fit nearest neighbors model on the unlabeled data
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(self.X_unlabeled)

        # Find the nearest neighbors in the unlabeled data for each row in the labeled data
        distances, indices = nn.kneighbors(self.X_labeled)                         # indices is 2D matrix: (self.X_labeled.shape[0], self.X_unlabeled.shape[0])

        # Initialize the probability matrix
        prob_matrix = np.zeros((self.X_labeled.shape[0], self.X_unlabeled.shape[0]))

        # Fill the probability matrix based on nearest neighbors
        for i, neighbors in enumerate(indices):
            prob = 1.0 / len(neighbors)
            for neighbor in neighbors:
                prob_matrix[i, neighbor] = prob

        return convert_matrix_to_policy(prob_matrix)


def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P
