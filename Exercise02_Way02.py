import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df1 = pd.read_csv("data/data/optics/opt.trn", header=None)
df2 = pd.read_csv("data/data/optics/opt.tst", header=None)
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



X_train, y_train = df1.iloc[:, :-1], df1.iloc[:, -1]
X_test, y_test = df2.iloc[:, :-1], df2.iloc[:, -1]


knn_classification(X_train, y_train, X_test, y_test, k=1)


knn_classification(X_train, y_train, X_test, y_test, k=3)