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
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os
from copy import deepcopy

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

#read MAE and RMSE files
readData = pd.read_csv('new_results/ModelsInfo25x25_modified.csv', header=None)
temp = np.array(readData[3])[1:]
print(temp)

def data_visualization_2dr(w_data, title, i=0, visualize=True):
    if visualize:
        plt.axis([0, len(w_data[0]), 0, len(w_data)])
        w_data[w_data < 0] = 0
        # w_data[w_data >= 100] = 0
        x, y = w_data.nonzero()
        # x = range(0, 65)
        # y = range(0, 44)
        c = w_data[x, y]
        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        plt.title(title)
        plt.colorbar()
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()
#
f_array = []
f_index = 0
for grid_y in range(1, 45): # for every y
    for grid_x in range(1, 66): # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
        if not tempCheck.any():
            f_array.append(0)
        else:
            f_array.append(temp[f_index])
            f_index += 1

np.savetxt('new_results/reshapingIfLR.csv', f_array, delimiter=',', fmt='%s')
print(np.array(f_array).reshape((44, 65)))

# #read MAE and RMSE files
# readData = pd.read_csv('new_results/reshaping25x25mae.csv', header=None)
# temp = pd.to_numeric(np.array(readData[0])[:]).reshape((44, 65))
# data_visualization_2dr(temp, title='MAE')

