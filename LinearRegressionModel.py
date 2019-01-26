# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 01:00:43 2018

Unworking with leafsdata but works with salarydata - needs updating

@author: Dylan
"""


# simple linear regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Importing the dataset
dataset = pd.read_csv('leafsdata.csv') # change

# matrix of features / independantvariables
X = dataset.iloc[:, 20].values # index in column
# vector of the dependant variable
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
# possibly change test size
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
                                                    random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # independant and dependent var

# predicting test set results
# vector of predictions of the dependant variable
y_pred = regressor.predict(X_test)

# visualizing/plotting training data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing/plotting test data
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
