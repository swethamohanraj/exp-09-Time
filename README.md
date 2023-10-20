# EXP-9 POLYNOMIAL-TREND-ESTIMATION

## AIM:
Implementation of Polynomial Trend Estiamtion Using Python.

## ALGORITHM:
1) Import necessary libraries (NumPy, Matplotlib, Pandas, scikit-learn).
2) Load the dataset using Pandas.
3) Extract features (independent variable) and the target variable from the dataset.
4) Choose the degree of the polynomial for regression.
5) Use PolynomialFeatures to transform the features into polynomial features of the chosen degree.
6) Fit a linear regression model to the polynomial features.
7) Visualize the original data points and the polynomial regression curve.
8) Optional: If needed, predict the target variable for new data points using the trained model.
9) End the program.

## PROGRAM:
### A - LINEAR TREND ESTIMATION
```python
def calculateB(x, y, n):
 
 # sum of array x 
  sx = sum(x)
 # sum of array y 
  sy = sum(y)
 
 # for sum of product of x and y 
  sxsy = 0
 # sum of square of x 
  sx2 = 0
  for i in range(n):
      sxsy += x[i] * y[i]
      sx2 += x[i] * x[i]
  b = (n * sxsy - sx * sy)/(n * sx2 - sx * sx)
  return b

def leastRegLine(X,Y,n):
 
 # Finding b 
  b = calculateB(X, Y, n)
  meanX = int(sum(X)/n)
  meanY = int(sum(Y)/n)
  # Calculating a
  a = meanY - b * meanX
  # Printing regression line 
  print("Regression line:")
  print("Y = ", '%.3f'%a, " + ", '%.3f'%b, "*X", sep="")
  return a,b

X = [95, 85, 80, 70, 60 ]
Y = [90, 80, 70, 65, 60 ]
n = len(X)
a,b=leastRegLine(X, Y, n)

for i in range(len(X)):
  Y[i]=a+b*X[i]
  print("%.3f"%Y[i])


### B - POLYNOMIAL REGRESSION
python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
datas = pd.read_csv('data.csv')
datas
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values
# Features and the target variables
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y)
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
# Visualising the Linear Regression results
plt.scatter(X, y, color='blue')
plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()
plt.scatter(X, y, color='blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)),
 color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()
# Predicting a new result with Linear Regression
# after converting predict variable to 2D array
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)
# Predicting a new result with Polynomial Regression
# after converting predict variable to 2D array
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
```

## OUTPUTS
### A - LINEAR TREND ESTIMATION
<img width="149" alt="image" src="https://github.com/Monisha-11/POLYNOMIAL-TREND-ESTIMATION/assets/93427240/a7af1bfe-58c9-4c8d-aa59-893718b6c8ce">

### B - POLYNOMIAL REGRESSION
<img width="281" alt="image" src="https://github.com/Monisha-11/POLYNOMIAL-TREND-ESTIMATION/assets/93427240/f2f93fff-50dc-406d-b9c6-443f57f297f6">

## RESULT:

Thus the program run successfully based on the Polynomial Trend Estiamtion model.
