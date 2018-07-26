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

# read MAE and RMSE files
readData = pd.read_csv('new_results/RMSE25x25_calculations_modified.csv', header=None)

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

def show_images(images, cols, titles):
    min_v = np.nanmin(images)
    max_v = np.nanmax(images[images != np.inf])
    print(min_v, max_v)
    # assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
    fig.suptitle(titles)
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(6, 4, n + 1)
        plt.axis([0, len(image[0]), 0, len(image)])
        # image[image >= 0] = 0
        # image[image > 10] = 0
        x, y = image.nonzero()
        c = image[x, y]

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=1)
        if n == 8:
            plt.ylabel('Vertical Grid')
        if n == 21:
            plt.xlabel('Horizontal Grid')
        plt.clim(min_v, max_v)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    plt.savefig('old_lowest_error_per_model.png')

imagesArr = []
for i in range(2, 26):
    print(i)
    oldErr = pd.to_numeric(np.array(readData[i])[1:])
    oldBest = pd.to_numeric(np.array(readData[26])[1:])
    oldErr[oldErr != oldBest] = 0
    imagesArr.append(oldErr)

final_lists = []
for list in imagesArr:
    f_array = []
    f_index = 0
    for grid_y in range(1, 45):  # for every y
        for grid_x in range(1, 66):  # for every x
            print('=================PLACE:', grid_x, grid_y, '=====================')
            tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
            if not tempCheck.any():
                f_array.append(0)
            else:
                f_array.append(list[f_index])
                f_index += 1
    f_array = np.array(f_array).reshape((44, 65))
    final_lists.append(f_array)

show_images(final_lists, 1, titles="Lowest Root Mean Square Error for each old model")
