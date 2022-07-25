import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

data = pd.read_csv("HepatitisCdata.csv")
print(data.head())

data['Category'] = data['Category'].replace(['0=Blood Donor'],'0')
data['Category'] = data['Category'].replace(['1=Hepatitis'],'1')
data['Category'] = data['Category'].replace(['2=Fibrosis'],'2')
data['Category'] = data['Category'].replace(['3=Cirrhosis'],'3')
print(data.head())
data['Category'] = pd.to_numeric(data['Category'],errors = 'coerce')
data['Sex'] = data['Sex'].replace(['m'],'1')
data['Sex'] = data['Sex'].replace(['f'],'0')
data['Sex'] = pd.to_numeric(data['Sex'],errors = 'coerce')

data = data.dropna()
print(data.isnull().sum())
print(data.dtypes)
X = data.drop("Category", axis=1)
std_data = StandardScaler()
std_data.fit(X)
X = std_data.transform(X)
Y = data['Category']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print("the accuracy percentage  with Svm is ", accuracy*100 , "%")


joblib.dump(model,'model_joblib')
load_model = joblib.load('model_joblib')

accuracy_2 = accuracy_score(Y_test,load_model.predict(X_test))
print(accuracy_2)

import pickle

with open('model_pickel', 'wb') as file:
    pickle.dump(model,file)

with open('model_pickel', 'rb') as file:
    pickle_model = pickle.load(file)

Y_pred_pickel = pickle_model.predict(X_test)
accuracy_3 = accuracy_score(Y_test,Y_pred_pickel)
print(accuracy_3)