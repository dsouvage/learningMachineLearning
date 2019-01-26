# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 01:35:20 2018

@author: Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv') # use pandas to read data set

X = dataset.iloc[:, :-1].values # take all the values in the data set except 
# for the last one
y = dataset.iloc[:, 3].values # take final column of data set

# dealing with missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) # 3 = upperbound not included thus 0-2 = 1-3 
X[:,1:3] = imputer.transform(X[:,1:3])

# encoding categorical data and then categorizing it
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0])
ohenc = OneHotEncoder(categorical_features = [0])
X = ohenc.fit_transform(X).toarray()
# turns Yes into 1, No into 0
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# splitting from dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
# for X
sscX = StandardScaler() #important to fit X train before X test
X_train = sscX.fit_transform(X_train) # need to fit and transform
X_test = sscX.transform(X_test) # already fitted

# scaling dummy variables = based on interpration of the models
# losing interpretation belongs to which "country"
# scaling dummy = more accuracy in prediction though

sscY = StandardScaler() # we dont need to apply because 
# it is a classification problem with a categorical variable
# however for regression when the dependant variable will take a huge 
# range of values we will have to