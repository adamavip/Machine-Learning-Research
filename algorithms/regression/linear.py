import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    """
    Linear Regression algorithm

    Training:
        - Initialize weight as zero (or a value between 0 and 1)
        - Initialize bias as zero

    Given a data point:
        - Predict result using y_hat = wx +b
        - Calculate the error 
        - Use gradient descent to determine new weight and bias values
        - Repeat n times
    """

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = None
        self.bias = None

        for _ in range(self.n_iters):
            self.weights = np.zeros(n_features)
            self.bias = 0
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
