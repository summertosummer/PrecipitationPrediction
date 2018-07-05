import pickle
import tensorflow as tf
from netCDF4 import Dataset
import pickle
import numpy as np
import csv
import Orange as og
import math
import itertools as it
from Orange.data import Domain, Table
import random
from Orange.projection import PCA
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

with open('random70.csv') as csvf:
    ind70 = csv.reader(csvf)
    indexi70 = list(ind70)
    index70 = indexi70[0]

with open('random30.csv') as csvf:
    ind30 = csv.reader(csvf)
    indexi30 = list(ind30)
    index30 = indexi30[0]

# #read MAE and RMSE files
# dfMAE = pd.read_csv('MAE25x25.csv', header=None)
# dfRMSE = pd.read_csv('RMSE25x25.csv', header=None)


# creating the whole dataset (inputs and target), not dividing into training and testing here
def create_training_data(grid_x, grid_y):
    data_x = [] # inputs
    data_y = [] # target
    tr_count = 0
    for i in index70: # working with 20 days
        for j in range(10): # 10 times in each day
            x = []
            for k in range(1, 25): # 24 models as input
                # print('model: ', k)
                b = rain_models[i, j, k, grid_y - 1:grid_y + 2, grid_x - 1:grid_x + 2] #taking an area of 9X9 from every model
                rain100 = np.array(b)
                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, grid_y, grid_x] #taking the real data as target, zero in the third dimention is for real data
            rainR = bt

            data_y.append(rainR) # appending real rain data
            data_x.append(list(it.chain.from_iterable(x))) # appending inputs

    return data_x, data_y

# creating the whole dataset (inputs and target), not dividing into training and testing here
def create_testing_data(grid_x, grid_y):
    data_x = [] # inputs
    data_y = [] # target
    tr_count = 0
    for i in index30: # working with 20 days
        for j in range(10): # 10 times in each day
            x = []
            for k in range(1, 25): # 24 models as input
                # print('model: ', k)
                b = rain_models[i, j, k, grid_y - 1:grid_y + 2, grid_x - 1:grid_x + 2] #taking an area of 9X9 from every model
                rain100 = np.array(b)
                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, grid_y, grid_x] #taking the real data as target, zero in the third dimention is for real data
            rainR = bt

            data_y.append(rainR) # appending real rain data
            data_x.append(list(it.chain.from_iterable(x))) # appending inputs

    return data_x, data_y

# dividing into training and testing dataset here
# training new models using the training data
# storing the new models
# predicting target for the testing data
# calculating mae and rmse of the new prediction
def run_models(grid_y, grid_x):
    X_train, Y_train = create_training_data(grid_x, grid_y) # X and Y is the inputs and target
    data = Table(X_train, Y_train) # creating a Orange table combining both X and Y

    feature_method = og.preprocess.score.UnivariateLinearRegression() # feature selection
    selector = og.preprocess.SelectBestFeatures(method=feature_method, k=50) # taking 50 features out of 216
    out_data2 = selector(data) # this is the new dataset with 50 features

    pca = PCA(n_components=5) # PCA with 5 components
    model = pca(out_data2)
    train2 = model(out_data2)

    featuresIndex = set()
    for comp in range(len(model.components_)-1, 0, -1):
        top2 = (- np.array(model.components_[comp])).argsort()[:2]
        featuresIndex |= set(top2)

    top2 = (- np.array(model.components_[0])).argsort()[:13]
    f_index = 0
    while(len(featuresIndex) != 13):
        featuresIndex.add(top2[f_index])
        f_index += 1

    ind = np.array(list(featuresIndex))

    # train = Table(list(out_data2[:,ind]), Y_train)
    # print(train)
    store = np.array(pca.domain)[ind]
    # print(store)
    np.savetxt('unlucky13/' + str(grid_x) + '_' + str(grid_y) + '.csv', store, delimiter=',', fmt='%s')

total = 0
countMAE = 0
countRMSE = 0
tempCheck = []
for grid_y in range(1, 45): # for every y
    for grid_x in range(1, 66): # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
        if not tempCheck.any():
            continue
        run_models(grid_y, grid_x)
