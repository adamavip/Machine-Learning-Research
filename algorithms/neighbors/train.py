from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.model_selection import train_test_split

iris = load_iris()
print(iris.keys())
X, y = iris.data, iris.target
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# Load the KNN classifier and train the model
clf = KNN(K=5)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
# print(predictions)

# Compute the accuracy
acc = sum(predictions == y_test) / len(y_test)
print(acc)  # -> 0.966
