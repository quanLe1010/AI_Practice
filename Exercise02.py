import numpy as np
# Load the data
file1_path = "data/data/optics/opt.trn"
file2_path = "data/data/optics/opt.tst"

df1 = np.genfromtxt(file1_path, delimiter=',', dtype=float)
df2 = np.genfromtxt(file2_path, delimiter=',', dtype=float)

# Define the Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# KNN classification function
def knn_classification(X_train, y_train, X_test, k):
    y_pred = []

    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        nearest_indices = np.argsort(distances)[:k]  # Get indices of k-nearest neighbors
        nearest_labels = y_train[nearest_indices]  # Get labels of k-nearest neighbors
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]  # Majority vote for the label
        y_pred.append(predicted_label)

    return np.array(y_pred)


X_train, y_train = df1[:, :-1], df1[:, -1].astype(int)
X_test, y_test = df2[:, :-1], df2[:, -1].astype(int)


# Define a function to calculate the confusion matrix
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        true_label = y_true[i]
        predicted_label = y_pred[i]

        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix


# Perform KNN classification for k=1 and k=3
k_values = [1, 3]

for k in k_values:
    y_pred = knn_classification(X_train, y_train, X_test, k)
    accuracy = np.mean(y_pred == y_test)

    print(f"KNN Classification (k={k}):")
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate and print the confusion matrix
    num_classes = len(np.unique(y_test))
    cm = calculate_confusion_matrix(y_test, y_pred, num_classes)
    print(f"Confusion Matrix (k={k}):")
    print(cm)
