from algorithms.regression.linear import LinearRegression
import numpy as np

lr = LinearRegression()
print(lr)
X = list(np.random.randint(2, 9, size=10))
y = [1.5*j+2 for j in X]
lr.fit(X, y)
