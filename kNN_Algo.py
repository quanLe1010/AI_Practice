import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

data = pd.read_csv('iris_1D.csv')

# Get x
x_data = data['Petal_Length'].to_numpy()
# trích xuất cột Petal_Length và chuyển nó thành 1 mảng numpy
x_data = x_data.reshape(6,1)
#print(x_data)

# Get y
y_data = data['Label'].to_numpy()
#print(y_data)

x_test = [[2.4]]

classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(x_data, y_data)

y_pred = classifier.kneighbors(x_test)
classifier.classes_[classifier._y[3]]
print(y_pred)
