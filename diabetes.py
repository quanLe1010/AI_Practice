import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetes.csv')
# print(len(data))
# print(data.head(n = 3))
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    data[column] = data[column].replace(0,np.NaN)
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.NaN, mean)

#print(data['Glucose'])
X = data.iloc[:, 0:8]
y = data.iloc[:,8]
X_train, X_test, y_train,y_test = train_test_split(X,y,random_state = 0, test_size=0.2)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='euclidean',metric_params=None,n_jobs=1, n_neighbors=11,p=2,weights='uniform')
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))