import inline as inline
import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
data = load_digits()

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print(dir(data))

x = data.data
y = data.target
print(data.images[0])

plt.matshow(data.images[0])
plt.show()

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(len(x_train))
print(len(x_test))

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(model.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))