# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# """remove last column"""
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
#0 = cali, 1 = florida, 2 = ny
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:, 3] = labelEncoderX.fit_transform(X[:, 3])
ohenc = OneHotEncoder(categorical_features = [3])
X = ohenc.fit_transform(X).toarray()

#dont need to do this
# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)


#avoiding the dummy variable trap
#remove the first column from X, dont take index 0
#note the lib does it for us but we can do it manually
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting multiple Lin Reg to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting test results
y_pred = regressor.predict(X_test)
