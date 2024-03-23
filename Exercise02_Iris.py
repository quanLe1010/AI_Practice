import numpy as np
import pandas as pd
import math

# Đọc tập huấn luyện từ tệp CSV
data_train = pd.read_csv('data/data/iris/iris.trn',header=None)

# Đọc tập kiểm tra từ tệp CSV
data_test = pd.read_csv('data/data/iris/iris.tst',header=None)
print(data_test)
# print(data_train.shape)
feature_columns = [0,1,2,3]
X1 = data_train[feature_columns].values
y1 = data_train[4].values

X2 = data_test[feature_columns].values
y2 = data_test[4].values
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def get_neighbors(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((X_train[i], y_train[i], dist))

    # Sort the distances and get the k-nearest neighbors
    distances.sort(key=lambda x: x[2])  # Sort by distance
    neighbors = distances[:k]

    return neighbors


# Predict the class for a test data point using k-nearest neighbors
def predict_classification(X_train, y_train, test_point, k):
    neighbors = get_neighbors(X_train, y_train, test_point, k)
    class_votes = {}  # Dictionary to store class votes

    for neighbor in neighbors:
        _, neighbor_class, _ = neighbor
        if neighbor_class in class_votes:
            class_votes[neighbor_class] += 1
        else:
            class_votes[neighbor_class] = 1

    # Find the class with the most votes
    predicted_class = max(class_votes, key=class_votes.get)

    return predicted_class


# Example usage:
k = 5  # Number of neighbors to consider
test_data_point = X2[3]  # Use the first data point from the test set

predicted_class = predict_classification(X1, y1, test_data_point, k)
# print(f"Predicted Class: {predicted_class}")

def calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / float(len(y_true))

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for i in range(len(y_true)):
        true_class = y_true[i]
        predicted_class = y_pred[i]
        confusion_matrix[true_class][predicted_class] += 1
    return confusion_matrix


k_values = [1, 3, 5]  # Add more k values as needed

for k in k_values:
    y_pred = [predict_classification(X1, y1, test_point, k) for test_point in X2]

    accuracy = calculate_accuracy(y2, y_pred)
    num_classes = len(np.unique(y1))
    confusion_matrix = calculate_confusion_matrix(y2, y_pred, num_classes)

    print(f"k={k}, Accuracy: {accuracy}")
    print("Confusion Matrix:")
    for row in confusion_matrix:
        print(row)