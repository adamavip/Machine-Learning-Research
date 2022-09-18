import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
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

    def __init__(self) -> None:
        self.theta = np.random.random(2)

    def __repr__(self) -> str:
        return f"{self.theta[0]:.5f} + {self.theta[1]:.5f} * X"

    @staticmethod
    def assert_sizes(*arrays):
        default_len = len(arrays[0])
        for array in arrays:
            if len(array) != default_len:
                raise Exception("Arrays should have same size")

    def hypothesis(self, x):
        """Linear regression hypothesis : h(Ø) = Ø_o + Ø_1 * x"""
        return self.theta[0] + self.theta[1] * x

    @staticmethod
    def cost(y_true, y_pred):
        """
            Cost function of linear regression J(Ø) = sum(y_pred - y_true)^2
        """
        cost = 0
        for i in range(len(y_true)):
            cost += pow((y_true[i]-y_pred[i]), 2)

        cost /= len(y_true)

        return cost

    def _gradient_descent(self, m, b, X, y, alpha=0.001):
        m_gradient = 0.0
        b_gradient = 0.0

        # Compute the partial gradients with respect to m and b
        for i in range(len(X)):
            m_gradient += (y[i] - self.hypothesis(X[i])) * X[i]
            b_gradient = (y[i] - self.hypothesis(X[i]))

        m_gradient = (-2 * m_gradient) / len(X)
        b_gradient = (-2 * b_gradient) / len(X)

        # Update the parameters
        m = m - alpha * m_gradient
        b = b - alpha * b_gradient

        return m, b

    def fit(self, X, y, lr=0.01, epochs=100, verbose=True, visualize=True):
        """Linear regression training method, fits function to data"""
        self.assert_sizes(X, y)

        # Determine when to output the cost values
        if verbose:
            consistency = int(epochs / 20)

        # Iterate over epochs
        for epoch in range(epochs):
            self.theta[1], self.theta[0] = self._gradient_descent(self.theta[1],
                                                                  self.theta[0],
                                                                  X, y,
                                                                  alpha=lr)
            # Compute the cost function
            y_hat = [self.hypothesis(X[i]) for i in range(len(X))]
            cost = self.cost(y, y_hat)
        if verbose:
            consistency = int(epochs / 20)
            if (int(epoch % 10) == 0):
                print("Epoch: {} Cost: {}".format(epoch, cost))

        # Visualize training data
        if visualize:
            # Compute model output with best parameters
            output = [self.hypothesis(X[i]) for i in range(len(X))]

            # Visualize true vs predictions for calibration
            plt.title('Training')
            plt.scatter(X, y)
            plt.plot(X, output, color='red')
            plt.show()

    def predict(self, X_test):
        """Use best estimated parameters to make predictions"""
        return [self.hypothesis(X_test[i]) for i in range(len(X_test))]
