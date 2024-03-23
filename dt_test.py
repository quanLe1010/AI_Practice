import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

data_train = [
    ["sunny", 85, 85, False, "Don't Play"],
    ["sunny", 80, 90, True, "Don't Play"],
    ["overcast", 83, 78, False, "Play"],
    ["rain", 70, 96, False, "Play"],
    ["rain", 68, 80, False, "Play"],
    ["rain", 65, 70, True, "Don't Play"],
    ["overcast", 64, 65, True, "Play"],
    ["sunny", 72, 95, False, "Don't Play"],
    ["sunny", 69, 70, False, "Play"],
    ["rain", 75, 80, False, "Play"],
    ["sunny", 75, 70, True, "Play"],
    ["overcast", 72, 90, True, "Play"],
    ["overcast", 81, 75, False, "Play"],
    ["rain", 71, 80, True, "Don't Play"],
]
df_train = pd.DataFrame(data_train, columns=["Outlook", "Temperature", "Humidity", "Windy", "Class"])
# Tạo biến LabelEncoder cho từng cột
le_outlook = LabelEncoder()
le_windy = LabelEncoder()
le_class = LabelEncoder()

# Huấn luyện bộ mã hóa cho từng cột
le_outlook.fit(df_train["Outlook"])
le_windy.fit(df_train["Windy"])
le_class.fit(df_train["Class"])

# Mã hóa dữ liệu và gán lại cho các cột
df_train["Outlook"] = le_outlook.transform(df_train["Outlook"])
df_train["Windy"] = le_windy.transform(df_train["Windy"])
df_train["Class"] = le_class.transform(df_train["Class"])

# print(df_train)
# print(df_train)
X_train = df_train.iloc[:, :-1].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()

data_test = [["overcast", 63, 70, False],
          ["rain", 73, 90, True],
          ["sunny", 70, 73, True]]
df_test = pd.DataFrame(data_test, columns=["Outlook", "Temperature", "Humidity", "Windy"])
# Huấn luyện bộ mã hóa cho từng cột
le_outlook.fit(df_test["Outlook"])
le_windy.fit(df_test["Windy"])

# Mã hóa dữ liệu và gán lại cho các cột
df_test["Outlook"] = le_outlook.transform(df_test["Outlook"])
df_test["Windy"] = le_windy.transform(df_test["Windy"])
X_test = df_test.to_numpy()
# print(X_test)
# print(y_train)
# print(df_test)

# print(X_train)

dt_classifier = DecisionTreeClassifier(max_depth=5,random_state=42, criterion='entropy')
dt_classifier.fit(X_train,y_train)
y_pred = dt_classifier.predict(X_test)
print(y_pred)