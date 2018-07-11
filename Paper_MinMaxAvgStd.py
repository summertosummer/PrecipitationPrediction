'''
Before we do work on these
Let us create plots for the old models

These plots
Email them to me
And include in the latex

In a new section - for now call it appendix

I think rmse is doing well - if you agree, from now on let’s work only with rmse
1. Send me the rainfall info
      Min and max overall
      Avg overall
      Steve
      Min and max when u sum for 25 as u have done so far
      Avg for this
      Steve

      Plots for min, max, avg and Stdev for the 25

2. 24 plots - Error for all the models
3. 24 *4 plots -error  for all the model when prediction
Falls within time interval midnight-6 am, 6 am- noon, noon-6 pm and 6-midnight
'''

import numpy as np
import csv
from netCDF4 import Dataset
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

# SOLVING PROBLEM 1
def calculation():
    #reading netcdf
    netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
    rain_models = netcdf_entire_dataset.variables['summing_models']

    minmaxArray = []
    for y in range(46): # 46 y-coordinates
        for x in range(67): # 67 x-coordinates
            # for i in range(1, 25):  # for every model
            print(y, x)
            temp = []
            data = np.array(rain_models[:20, :10, 0, y, x])
            temp.append(np.nanmax(data))
            temp.append(np.nanmin(data))
            temp.append(np.nanmean(data))
            temp.append(np.nanstd(data))

            minmaxArray.append(temp)

    np.savetxt('minMaxAvgStd.csv', minmaxArray, delimiter=',', fmt='%10.5f')


#2D Visualizaiton
def data_visualization_2dr(w_data, title, i=0, visualize=True):
    if visualize:
        plt.axis([0, len(w_data[0]), 0, len(w_data)])
        # w_data[w_data >= 0] = 0
        # w_data[w_data >= 100] = 0
        x, y = w_data.nonzero()
        # x = range(0, 65)
        # y = range(0, 44)
        c = w_data[x, y]
        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        plt.title(title)
        plt.colorbar()
        plt.ylabel('Vertical Grid')
        plt.xlabel('Horizontal Grid')
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()

readData = pd.read_csv('minMaxAvgStd.csv', header=None)
temp1 = pd.to_numeric(np.array(readData[3])[:]).reshape((46, 67))
temp2 = pd.to_numeric(np.array(readData[2])[:]).reshape((46, 67))
temp = temp1 / temp2
data_visualization_2dr(w_data=temp, title='Standard Deviation of Precipitation For each grid point divided by average rainfall')