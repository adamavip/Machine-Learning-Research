from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from linear import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=5, noise=5, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

#print(X_train.shape, y_train.shape)


reg = LinearRegression(lr=0.8)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = np.mean((y_test-predictions)**2)
print('MSE: ', mse)
print('R2: ', r2)
""" plt.figure()
plt.scatter(X_test[:, 2], y_test, cmap='jet')
plt.plot(X_test[:, 2], predictions, color='red')
plt.show() """

# Hyperparameter optimization
""" lrs = [0.1, 0.2, 0.4, 0.8]
mses = []

for lr in lrs:
    reg = LinearRegression(lr=lr)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    mse = np.mean((y_test-predictions)**2)
    mses.append(mse)

plt.plot(lrs, mses, 'o-')
plt.show() """
