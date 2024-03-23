from random import seed, randrange
from csv import reader



def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset



def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup



def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split



def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for _ in range(len(train[0]))]
    for _ in range(n_epoch):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights



def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
def confusion_matrix(actual, predicted):
    unique_classes = set(actual)
    matrix = [[0 for _ in range(len(unique_classes))] for _ in range(len(unique_classes))]
    class_index = {clazz: i for i, clazz in enumerate(unique_classes)}

    for i in range(len(actual)):
        row = class_index[actual[i]]
        col = class_index[predicted[i]]
        matrix[row][col] += 1

    return matrix

def print_confusion_matrix(matrix):
    print("Confusion Matrix:")

    for i in range(len(matrix)):
        print(i, end="\t\t")
        for j in range(len(matrix[0])):
            print(matrix[i][j], end="\t")
        print()

# Modify the code to obtain actual and predicted values
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    all_actual = []
    all_predicted = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores, all_actual, all_predicted
seed(1)
trainset_filename = input("Enter the trainset: ")
testset_filename = input("Enter the testset: ")
l_rate = float(input("Enter the learning rate: "))
maxit = int(input("Enter the maximum number of epochs: "))
n_folds = 3


trainset = load_csv(trainset_filename)
for i in range(len(trainset[0]) - 1):
    str_column_to_float(trainset, i)

str_column_to_int(trainset, len(trainset[0]) - 1)


testset = load_csv(testset_filename)
for i in range(len(testset[0]) - 1):
    str_column_to_float(testset, i)

str_column_to_int(testset, len(testset[0]) - 1)


scores, all_actual, all_predicted = evaluate_algorithm(trainset, perceptron, n_folds, l_rate, maxit)

print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
conf_matrix = confusion_matrix(all_actual, all_predicted)
print_confusion_matrix(conf_matrix)