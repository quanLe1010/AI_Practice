import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('trains.txt', delim_whitespace=True)
#print(data)
x_test = np.array([[0.550000, 0.364000], [0.558000, 0.470000], [0.456000, 0.450000], [0.450000, 0.570000]])
x_train = data[['X1','X2']].to_numpy()
#x_test = np.array([0.550000, 0.364000])
y = data['Class'].to_numpy()
# print(x)
# print(y)
def euclidean_distance(x_train, x_test):
    distance = []
    d = 0.0
    for i in x_test:
        x_test_extended = np.tile(i, (len(x_train), 1))
        d = np.sqrt(np.sum((x_train - x_test_extended)**2, axis=1))
        distance.append(d)
    return np.array(distance)

result = euclidean_distance(x_train, x_test)
print(result)

K = 3
z = sorted(zip(result.tolist(), y.tolist()))[:K]
print(z)