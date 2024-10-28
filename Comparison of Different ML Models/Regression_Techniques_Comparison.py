import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Students.csv')
x = df.iloc[:,0:1].values
y = df.iloc[:,1].values

plt.title('raw data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x, y, color='red')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

plt.title('linear Regression on training data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, linear_model.predict(x_train), color='blue')
plt.show()

plt.title('Predicting on testing data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, linear_model.predict(x_test), color='blue')
plt.show()

slope = linear_model.coef_
y_intercept = linear_model.intercept_
R_square_linear = linear_model.score(x, y)
mean_square_linear = mean_squared_error(y, linear_model.predict(x))


poly_model = PolynomialFeatures(degree = 14)
x_poly = poly_model.fit_transform(x_train)

linear_model_pol = LinearRegression()
linear_model_pol.fit(x_poly, y_train)

x_sample  = np.arange(min(x_train), max(x_train), 0.1)
x_sample = x_sample.reshape(len(x_sample), 1)

plt.title('Polynomial Regression on training data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x_train, y_train, color='red')
plt.plot(x_sample, linear_model_pol.predict(poly_model.fit_transform(x_sample)), color='blue')
plt.show()

x_sample_2  = np.arange(min(x_test), max(x_test), 0.1)
x_sample_2 = x_sample_2.reshape(len(x_sample_2), 1)

plt.title('Predicting on testing data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x_test, y_test, color='red')
plt.plot(x_sample_2, linear_model_pol.predict(poly_model.fit_transform(x_sample_2)), color='blue')
plt.show()

R_square_poly = linear_model_pol.score(poly_model.fit_transform(x), y)
mean_square_poly = mean_squared_error(y, linear_model_pol.predict(poly_model.fit_transform(x)))


decision_tree = DecisionTreeRegressor(random_state=0)
decision_tree.fit(x_train, y_train)

plt.title('Decision Tree on training data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, decision_tree.predict(x_train), color='blue')
plt.show()

plt.title('Predicting on testing data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, decision_tree.predict(x_test), color='blue')
plt.show()

R_square_tree = decision_tree.score(x, y)
mean_square_tree = mean_squared_error(y, decision_tree.predict(x))