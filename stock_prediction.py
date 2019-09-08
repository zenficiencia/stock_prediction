#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 06:56:25 2019

@author: Marco Zamora
"""

# stock prediction

# Importing the libraries
import quandl as ql               # stock data
import numpy as np                # array and math functions 
import matplotlib.pyplot as plt   # plotting

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


# Importing the dataset
dataset = ql.get("WIKI/SLB")
dataset = dataset[['Adj. Close']]

# Forecasting set up
forecast_out = int(30)            # predicting 30 days into future
dataset['Prediction'] = dataset[['Adj. Close']].shift(-forecast_out) # label column with data shifted

# Defining Features & Labels
X = np.array(dataset.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
y = np.array(dataset['Prediction'])
y = y[:-forecast_out]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# FITTING AND VISUALIZING REGRESSION MODELS

# Simple Linear Regressor
# Training
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Test
confidence = regressor.score(X_test, y_test)
# Predicting a new result
forecast_pred = regressor.predict(X_forecast)
print("Confidence: ", confidence)
print(forecast_pred)
#  Visualize the results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Stock Prediction (Simple Regression)')
plt.show()

# Support Vector Regressor
# Training
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
# Test
confidence = regressor.score(X_test, y_test)
# Predicting a new result
forecast_pred = regressor.predict(X_forecast)
print("Confidence: ", confidence)
print(forecast_pred)
#  Visualize the results
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Stock Prediction (Support Vector Regression)')
plt.show()

# Random Forest Regression
# Training
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X_train, y_train)
# Test
confidence = regressor.score(X_test, y_test)
# Predicting a new result
forecast_pred = regressor.predict(X_forecast)
print("Confidence: ", confidence)
print(forecast_pred)
#  Visualize the results
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Stock Prediction (Random Forest Regression)')
plt.show()

