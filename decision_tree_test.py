from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

from decision_tree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


def calculate_accuracy(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy


acc = calculate_accuracy(y_test, y_pred)
print(f'Classification accuracy = {acc}')