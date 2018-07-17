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

# read MAE and RMSE files
readData = pd.read_csv('25X25/ModelsInfo25x25.csv', header=None)
BestNew = np.array(readData[2])[1:]
# Poly = np.array(readData[29])[1:]
print(BestNew)

best2 = []
poly2 = []
f_index = 0
for grid_y in range(1, 45):  # for every y
    for grid_x in range(1, 66):  # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
        if not tempCheck.any():
            f_index += 1
        else:
            best2.append(BestNew[f_index])
            poly2.append(BestNew[f_index])
            f_index += 1

combine = []
combine.append(best2)
combine.append(poly2)

np.savetxt('MLModels.csv', np.array(combine).transpose(), delimiter=',', fmt='%s')