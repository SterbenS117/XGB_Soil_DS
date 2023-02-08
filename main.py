from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
#import openpyxl
import xgboost
from xgboost import XGBRegressor
from xgboost import DMatrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

resolution='XG_100m_SS_r2_1'
# train_file ='400m_processed_.csv'
# test_file ='400m_processed_.csv'
train_file = 'SS_100m_data_train.csv'
test_file = 'SS_100m_data_test.csv'

train_data = pd.read_csv(train_file, usecols=['SMERGE','Date','PageName','LAI','Albedo','NDVI','Clay','Sand','Silt ','Slope','Elevation','Ascept'])
train_data = train_data.loc[:, ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','LAI','Albedo']]
train_data.columns = ['Clay', 'Sand','Silt ', 'Elevation','Slope', 'Ascept','NDVI', 'SMERGE','Date','Lai','Albedo']

test_data = pd.read_csv(test_file, usecols=['SMERGE','Date','PageName','LAI','Albedo','NDVI','Clay','Sand','Silt ','Slope','Elevation','Ascept'])
test_data = test_data.loc[:, ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','LAI','Albedo']]
test_data.columns = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','Lai','Albedo']
page = pd.read_csv(test_file, usecols=['PageName'])
date = pd.read_csv(test_file, usecols=['Date'])

train_data['Date'] = pd.to_datetime(train_data['Date'], format="%m/%d/%Y").astype(int)
test_data['Date'] = pd.to_datetime(test_data['Date'], format="%m/%d/%Y").astype(int)

print(train_data.dtypes)
y_test = test_data[['SMERGE']]
x_test = test_data[['Clay','Sand', 'Elevation','Slope','NDVI', 'Lai', 'Albedo']]
y_train = train_data[['SMERGE']]
x_train = train_data[['Clay','Sand', 'Elevation','Slope','NDVI', 'Lai', 'Albedo']]

model = XGBRegressor(verbosity=1,n_estimators=500,max_depth=5,tree_method='hist')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

model.fit(x_train, y_train)

h = model.predict(x_test)
test_data['Date'] = date['Date']
test_data['ML_'] = h
test_data['PageName'] = page
test_data.to_csv(resolution + ".csv", index=False)
