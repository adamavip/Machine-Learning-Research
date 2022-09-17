import numpy as np
from collections import Counter


class KNN:
    """
    KNN algorithm

    Given a data point:
        - Compute its distance from all other data points in the dataset.
        - Get the K nearest points.
        - Regression: get the average of their values. 
        - Classification: get the label with majority vote.
    """

    def __init__(self, K):
        self.K = K

    @staticmethod
    def euclidian_distance(x1, x2):
        """Compute the euclidean distance between two records/data points"""
        return np.sqrt(np.sum((x1-x2)**2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the distances
        distances = [self.euclidian_distance(
            x, x_train) for x_train in self.X_train]
        # Get the indices of K closest points
        k_indices = np.argsort(distances)[:self.K]
        # Get the most common point
        counter = Counter(self.y_train[k_indices])
        most_common = counter.most_common()
        return most_common[0][0]
