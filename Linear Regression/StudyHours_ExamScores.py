
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('hours.csv')
x = df.iloc[:,1:2].values
y = df.iloc[:,2].values

plt.title('raw data')
plt.xlabel('study hours')
plt.ylabel('exam scores')
plt.scatter(x, y, color='red')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)

plt.title('linear model on training data')
plt.xlabel('study hours')
plt.ylabel('exam scores')
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, linear_model.predict(x_train), color='blue')
plt.show()

plt.title('linear model on testing data')
plt.xlabel('study hours')
plt.ylabel('exam scores')
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, linear_model.predict(x_test), color='blue')
plt.show()

exam_score = linear_model.predict([[6]])
r_square = linear_model.score(x, y)