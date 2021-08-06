# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# simple linear regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x, y)


# Polynomial Linear Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
reg_2 = LinearRegression()
reg_2.fit(x_poly, y)



# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)   # dividing x into small divisions to get smooth results
x_grid = x_grid.reshape((len(x_grid), 1)) # // // // // // // // //// // // .. .. // // // // //
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
reg.predict([[6.5]])    #330378.78787879 not that goood

# Predicting a new result with Polynomial Regression
reg_2.predict(poly_reg.fit_transform([[6.5]]))   # 158862.45265155 veru close to the told one