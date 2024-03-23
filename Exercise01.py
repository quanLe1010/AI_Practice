import numpy as np
import pandas as pd
from pandas import DataFrame
import math

data_train = [[0.376000,0.488000,0],[0.312000 ,0.544000, 0],[0.298000, 0.624000, 0],[0.394000, 0.600000, 0],[0.506000 ,0.512000, 0],[0.488000, 0.334000, 1],[0.478000, 0.398000, 1],[0.606000, 0.366000 , 1],[0.428000, 0.294000, 1],[0.542000, 0.252000, 1]]
data_test = [[0.550000, 0.364000], [0.558000, 0.470000],[0.456000, 0.450000],[0.450000, 0.570000]]

def euclidean_distance(row1,row2):
    d = 0.0
    length = len(row2) - 1
    for i in range(length):
        d += (row1[i] - row2[i])**2
    return math.sqrt(d)


def Get_Neighbors(train, test_row, num):
    distance = list()  # []
    data = []
    for i in train:
        dist = euclidean_distance(test_row, i)
        distance.append(dist)
        data.append(i)
    distance = np.array(distance)
    data = np.array(data)
    index_dist = distance.argsort() #in các PT theo thứ tự tăng dần(chỉ số mảng)

    data  = data[index_dist]
    #print(distance)
    neighbors = data[:num] # lấy ra một list gồm num phần tử đầu tiên từ danh sách data.

    return neighbors

neighbors = Get_Neighbors(data_train,data_train[1],5)
for neighbor in neighbors:
    print(neighbor)

data_train = np.array(data_train)  # Chuyển data_train thành một mảng NumPy để dễ dàng truy cập vào hàng cụ thể

# Lấy hàng dữ liệu của data_train[1]
test_row = data_train[1]
print("Test Row (data_train[1]):")
print(test_row)

# Tìm và hiển thị các neighbors của test_row
neighbors = Get_Neighbors(data_train, test_row, 5)
print("\nNeighbors:")
for neighbor in neighbors:
    print(neighbor)

def predict_classification(data_train, test_row, n_neighbors):
    Neighbors = Get_Neighbors(data_train, test_row, n_neighbors)
    Classes = []
    for i in Neighbors:
        Classes.append(i[-1]) # thêm nhãn của hàng dữ liệu hiện tại vào danh sách Classes
    prediction = max(Classes, key= Classes.count)
    return prediction

n_neighbors = 5
predictions = [predict_classification(data_train, test_row, n_neighbors) for test_row in data_test]

# Kết quả dự đoán cho từng hàng dữ liệu kiểm tra
print("Prediction: ",predictions)


