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

# #read MAE and RMSE files
# ifLR = pd.read_csv('new_results/reshapingIfLR.csv', header=None)
# checkIfLR = np.array(ifLR[0])[:].reshape((44, 65))
# checkIfLR = checkIfLR.astype(str)
# checkIfLR = np.char.replace(checkIfLR, " ", "")

# read MAE and RMSE files
readData = pd.read_csv('new_results/RMSE25x25_calculations_modified.csv', header=None)
# WAItself = pd.to_numeric(np.array(readData[29])[1:])
# BestNew = pd.to_numeric(np.array(readData[31])[1:])
#
# #read MAE and RMSE files
# readData11 = pd.read_csv('new_results/ModelsInfo25x25_modified.csv', header=None)
# isKNNLR = np.array(readData11[2])[1:]
# print(isKNNLR)

# read MAE and RMSE files
readData22 = pd.read_csv('final_results/ModelsInfo25x25_modified_final_calculation.csv', header=None)
WAItself = pd.to_numeric(np.array(readData22[19])[1:2391])
BestNew = pd.to_numeric(np.array(readData22[21])[1:2391])
isKNNLR = np.array(readData22[2])[1:2391]
print(isKNNLR)

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
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
    plt.savefig('n_knn_lr_wa.png')

imagesArr = []
for i in range(2, 26):
    print(i)
    oldErr = pd.to_numeric(np.array(readData[i])[1:])
    newErr = pd.to_numeric(np.array(readData[21])[1:2391])
    # newErr[newErr > oldErr] = 0
    temp_list = []
    for ind in range(len(oldErr)):
        if newErr[ind] <= oldErr[ind] and (isKNNLR[ind] == 'KNN' or isKNNLR[ind] == 'LR' or WAItself[ind] == BestNew[ind]):
            temp_list.append(1)
        else:
            temp_list.append(0)
    imagesArr.append(temp_list)

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


show_images(final_lists, 1, titles="Is best model linear in old models?")
# data_visualization_2dr(w_data=imagesArr[0], title='model')
# display_image(rmse)