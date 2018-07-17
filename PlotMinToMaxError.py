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
import collections

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

minMax = {}

for d in range(20):
    for t in range(10):
        for y in range(46):
            for x in range(67):
                print(d, t, y, x)
                temp = int(rain_models[d, t, 0, y, x])
                # pred = rain_models[d, t, 1:25, y, x]
                error = rain_models[d, t, 1:25, y, x] # pred - temp
                if temp in minMax:
                    print('here1')
                    data = minMax[temp]
                    newErr = (error + data) / 2
                    minMax[temp] = list(newErr)
                else:
                    minMax[temp] = list(error)

dictlist = []
minMax = collections.OrderedDict(sorted(minMax.items()))
keys = list(minMax.keys())
values = list(minMax.values())

# for k in range(len(keys)):
#     # print(keys[k])
#     dictlist.append(keys[k])
#     # dictlist.extend(values[k])

for key, value in zip(keys, values):
    temp = []
    temp.append(key)
    for val in value:
        temp.append(val)
    dictlist.append(temp)


#
# print(list(keys)[0])
# print(list(values)[0])

print(dictlist)
print(minMax)

np.savetxt('minToMax_pd.csv', np.array(dictlist), delimiter=',', fmt='%10.5f')
# np.savetxt('myCSV2.csv', np.array(minMax), delimiter=',', fmt='%10.5f')