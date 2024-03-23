import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
data_train = 'fp107/fp107/fp107.trn'
data_test = 'fp107/fp107/fp107.tst'
def changed(path):
    with open(path, 'r') as file:
        data = file.read()

    data_changed = data.replace(' ', ',')

    with open(path, 'w') as file_modified:
        file_modified.write(data_changed)

changed(data_train)
changed(data_test)

data_train = pd.read_csv('fp107/fp107/fp107.trn', header = None)
# print(data_train)
data_test = pd.read_csv('fp107/fp107/fp107.tst', header = None)
#print(data_test)
#data_train.info()
X_train = data_train.iloc[:, :-1].to_numpy()
y_train = data_train.iloc[:, -1].to_numpy()
X_test = data_test.iloc[:, :-1].to_numpy()
y_test = data_test.iloc[:, -1].to_numpy()

print(X_train)
model = GaussianNB()
# Model training
model.fit(X_train, y_train)
# Predict Output
predicted = model.predict([X_test[6]])

print("Actual Value: ", y_test[6])
print("Predicted Value: ", predicted[0])

# Calculate accuracy and F1-score
y_pred = model.predict(X_test)
accurate = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred,y_test, average="micro")
print("Confusion Matrix: \n", confusion_matrix(y_pred, y_test))
print("Accuracy: ", accurate)
print("F1 score: ", f1)
#print(metrics.classification_report(y_pred, y_test))
result = list(zip(y_pred,y_test))
print(result)
