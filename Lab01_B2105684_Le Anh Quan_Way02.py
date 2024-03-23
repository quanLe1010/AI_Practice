import pandas as pd
import numpy as np
import math
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


# print(data_train)
X_train = data_train.iloc[:, :-1].to_numpy()
y_train = data_train.iloc[:,-1].to_numpy()
X_test = data_test.iloc[:, :-1].to_numpy()
y_test = data_test.iloc[:,-1].to_numpy()
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

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
test_data_point = X_test[3]  # Use the first data point from the test set

predicted_class = predict_classification(X_train, y_train, test_data_point, k)
# print(f"Predicted Class: {predicted_class}")

def calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / float(len(y_true))

def calculate_confusion_matrix(y_true, y_pred):
    unique_classes = set(y_true + y_pred)
    num_classes = len(unique_classes)

    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    for i in range(len(y_true)):
        true_class = y_true[i]
        predicted_class = y_pred[i]
        confusion_matrix[true_class][predicted_class] += 1

    return confusion_matrix

def print_confusion_matrix(confusion_matrix):
    for row in confusion_matrix:
        print(" ".join(map(str, row)))

k_values = [1, 3, 5]

for k in k_values:
    y_pred = [predict_classification(X_train, y_train, test_point, k) for test_point in X_test]

    num_classes = len(np.unique(y_test))
    confusion_matrix = calculate_confusion_matrix(y_test, y_pred)
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"k={k}, Accuracy: {accuracy}")
    print(f"k={k}, Confusion Matrix:")
    print_confusion_matrix(confusion_matrix)
