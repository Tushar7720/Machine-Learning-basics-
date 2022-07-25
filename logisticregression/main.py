import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("HR_comma_sep.csv")

print(data.head())
data = data.drop(["Department"], axis=1)
print(data.head())

dummies = pd.get_dummies(data.salary)
print(dummies)

data = pd.concat([data, dummies], axis = 1)
print(data.head())
data = data.drop(['salary'], axis=1)
print(data.info())
print(data.isnull().sum())

data = data.drop(['low'], axis=1)
print(data.info())

X = data.drop(['left'],axis=1)
Y = data.left

std_data = StandardScaler()
std_data.fit(X)
X = std_data.transform((X))
print(X)
print(Y)

x_train ,x_test ,y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
A = accuracy_score(y_test,y_pred)
print(A)