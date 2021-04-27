# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the Dataset and observed X and Y values on the Dataset

dataSet = pd.read_csv('Position_Salaries.csv')
print("=========== Dataset =================")
print(dataSet)
print("X and Y values")
X = dataSet.iloc[:, 1:2].values
Y = dataSet.iloc[:, -1].values
print("X values = ", X)
print("Y values = ", Y)

# linear model
linearReg = LinearRegression()
linearReg.fit(X, Y)

# visualization of the results
plt.scatter(X, Y, color='red')
plt.plot(X, linearReg.predict(X), color='blue')
plt.title("Truth or Bluff(Linear)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Fitting a Polynomial Regression Model

# import PolynomialFeatures class. It is a transformer tool
# that transforms the matrix of features X into a new matrix of features X_poly

polyReg = PolynomialFeatures(degree=4)
X_Poly = polyReg.fit_transform(X)
linearReg1 = LinearRegression()
linearReg1.fit(X_Poly, Y)


# Visualization  polynomial results for higher resolution

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, linearReg1.predict(polyReg.fit_transform(X_grid)), color='blue')
plt.title("Truth or Bluff(Polynomial)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predict single observation
# double pair represent 2 D

print("predicted value for 6.5 experience with linear model ")
predict = linearReg.predict([[6.5]])
print("Salary of this person", predict)

print("predicted value for 6.5 experience with polynomial model ")
predict1 = linearReg1.predict(polyReg.fit_transform([[6.5]]))
print("Salary of this person", predict1)
