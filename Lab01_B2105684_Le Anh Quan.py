import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
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

data_train= pd.read_csv('fp107/fp107/fp107.trn',header=None)
data_test = pd.read_csv('fp107/fp107/fp107.tst', header=None)
def knn_classification(X_train, y_train, X_test, y_test, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Dự đoán nhãn trên tập kiểm tra
    predicted_labels = knn_model.predict(X_test)

    # Tính toán confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    # In confusion matrix
    print(f"\nConfusion Matrix (k={k}):")
    print(conf_matrix)

    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(y_test, predicted_labels)

    # In các chỉ số đánh giá
    print("Accuracy:", accuracy)



X_train, y_train = data_train.iloc[:, :-1], data_train.iloc[:, -1]
X_test, y_test = data_test.iloc[:, :-1], data_test.iloc[:, -1]


knn_classification(X_train, y_train, X_test, y_test, k=1)


knn_classification(X_train, y_train, X_test, y_test, k=3)

knn_classification(X_train, y_train, X_test, y_test, k=5)


