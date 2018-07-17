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
readData = pd.read_csv('final_results/ModelsInfo25x25_modified_final_calculation.csv', header=None)

def show_images(images, cols, titles):
    min_v = np.nanmin(images)
    max_v = np.nanmax(images[images != np.inf])
    print(min_v, max_v)
    # assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 4), dpi=100, facecolor='w', edgecolor='k')
    fig.suptitle(titles)
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(2, 4, n + 1)
        plt.axis([0, len(image[0]), 0, len(image)])
        # image[image >= 0] = 0
        # image[image > 10] = 0
        x, y = image.nonzero()
        c = image[x, y]

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=1)
        if n == 0:
            plt.ylabel('Linear Regression')
        if n == 1:
            plt.ylabel('K-Nearest Neighbors')
        if n == 2:
            plt.ylabel('Neural Network')
        if n == 3:
            plt.ylabel('Random Forest')
        if n == 4:
            plt.ylabel('Support Vector Machine')
        if n == 5:
            plt.ylabel('Polynomial Regression')
        if n == 6:
            plt.ylabel('Weighted Average')
        # if n == 8:
        #     plt.ylabel('Vertical Grid')
        # if n == 21:
        #     plt.xlabel('Horizontal Grid')
        plt.clim(min_v, max_v)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    plt.savefig('ML_error_among_new.png')

finalBestNew = []
for i in range(31, 38):
    BestNew = np.array(readData[i])[1:2391]
    finalBestNew.append(BestNew)

finalBestNew = np.array(finalBestNew)
print(finalBestNew[:, 1])

f_array = []
f_index = 0
for grid_y in range(1, 45):  # for every y
    for grid_x in range(1, 66):  # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
        if not tempCheck.any():
            f_array.append([0]*7)
        else:
            f_array.append(finalBestNew[:, f_index])
            f_index += 1

f_array = np.array(f_array)
# np.savetxt('bestNewML.csv', f_array, delimiter=',', fmt='%s')

final_arr = []
for i in range(7):
    temp = f_array[:, i]
    temp[temp == '0'] = '0.0001'
    temp[temp == ' '] = 0
    temp = temp.astype(np.float)
    # temp[temp > 0] = 1
    final_arr.append(temp.reshape((44, 65)))

show_images(np.array(final_arr), 1, titles="Error of each machine learning model performs better than other machine learning models")