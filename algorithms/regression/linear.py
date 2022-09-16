import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    """Simple linear regression implemented from scratch"""

    def __init__(self) -> None:
        self.theta = [0, 0]  # np.random.random(2)

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
        m = m + alpha * m_gradient
        b = b + alpha * b_gradient

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
            if (epoch % consistency == 0):
                print(f"Epoch: {epoch} Cost: {cost}")

        # Visualize training data
        if visualize:
            # Compute model output with best parameters
            output = [self.hypothesis(X[i]) for i in range(len(X))]

            # Visualize true vs predictions for calibration
            plt.scatter(X, y)
            plt.plot(X, output, color='red')
            plt.show()


lr = LinearRegression()
print(lr)
X = list(np.random.randint(2, 9, size=10))
y = [1.5*j+2 for j in X]
lr.fit(X, y)
