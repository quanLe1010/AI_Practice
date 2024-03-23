import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
data_train = 'data/data/fp/fp.trn'
data_test = 'data/data/fp/fp.tst'
def changed(path):
    with open(path, 'r') as file:
        data = file.read()

    data_changed = data.replace(' ', ',')

    with open(path, 'w') as file_modified:
        file_modified.write(data_changed)

changed(data_train)
changed(data_test)

data_train = pd.read_csv('data/data/fp/fp.trn', header = None)
# print(data_train)
data_test = pd.read_csv('data/data/fp/fp.tst', header = None)
#print(data_test)
#data_train.info()
X_train = data_train.iloc[:, :-1].to_numpy()
y_train = data_train.iloc[:, -1].to_numpy()
X_test = data_test.iloc[:, :-1].to_numpy()
y_test = data_test.iloc[:, -1].to_numpy()


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


dt_classifier = DecisionTreeClassifier(max_depth=12,random_state=42, criterion='entropy')
dt_classifier.fit(X_train,y_train)

# tree.plot_tree(dt_classifier)
# plt.show()

y_pred = dt_classifier.predict(X_test)
accurate = accuracy_score(y_test, y_pred)
f1 = f1_score(y_pred,y_test, average="micro")
print("Confusion Matrix: \n", confusion_matrix(y_pred, y_test))
print("Accuracy: ", accurate)
print("F1 score: ", f1)
result = list(zip(y_pred,y_test))
print(result)

