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

np.seterr(divide='ignore', invalid='ignore')

# def show_images(images, cols, titles):
#     min_v = np.nanmin(images)
#     max_v = np.nanmax(images[images != np.inf])
#     print(min_v, max_v)
#     # assert ((titles is None) or (len(images) == len(titles)))
#     n_images = len(images)
#     if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
#     fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
#     fig.suptitle(titles)
#     for n, (image, title) in enumerate(zip(images, titles)):
#         # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
#         a = fig.add_subplot(6, 4, n + 1)
#         plt.axis([0, len(image[0]), 0, len(image)])
#         # image[image >= 0] = 0
#         # image[image > 10] = 0
#         x, y = image.nonzero()
#         c = image[x, y]
#
#         im = plt.scatter(y[:], x[:], c=c[:], cmap='jet')
#         if n == 8:
#             plt.ylabel('Vertical Grid')
#         if n == 21:
#             plt.xlabel('Horizontal Grid')
#         plt.clim(min_v, max_v)
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     # plt.show()
#     plt.savefig('old_models_error.png')
#
# #read MAE and RMSE files
# readData = pd.read_csv('RMSE.csv', header=None)
#
# #read MAE and RMSE files
# readData2 = pd.read_csv('avgRainPerModel.csv', header=None)
#
# imagesArr = []
# for i in range(2, 26):
#     temp = np.array(readData[i])[1:]
#     temp[temp == ' '] = '0'
#     temp = pd.to_numeric(temp).reshape((46, 67))
#     imagesArr.append(temp)
#
# show_images(imagesArr, 1, titles="Root Mean Square Error of Different Prediction Models")


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

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        if n == 8:
            plt.ylabel('Vertical Grid')
        if n == 21:
            plt.xlabel('Horizontal Grid')
        plt.clim(min_v, max_v)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    plt.savefig('old_models_error_by_avg.png')

#read MAE and RMSE files
readData = pd.read_csv('RMSE.csv', header=None)

#read MAE and RMSE files
readData2 = pd.read_csv('avgRainPerModel.csv', header=None)

imagesArr = []
for i in range(2, 26):
    temp1 = np.array(readData[i])[1:]
    temp1[temp1 == ' '] = '0'
    temp1 = pd.to_numeric(temp1).reshape((46, 67))

    temp2 = pd.to_numeric(np.array(readData2[i-2])[:]).reshape((46, 67))
    print(temp2[0, 0])

    temp = temp1 / temp2
    print(temp)

    imagesArr.append(temp)

show_images(imagesArr, 1, titles="Root Mean Square Error of Different Prediction Models divided by average rainfall")

#
# def show_images(images, cols, titles):
#     # assert ((titles is None) or (len(images) == len(titles)))
#     n_images = len(images)
#     if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
#     fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
#     fig.suptitle(titles)
#     for n, (image, title) in enumerate(zip(images, titles)):
#         # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
#         a = fig.add_subplot(6, 4, n + 1)
#         plt.axis([0, len(image[0]), 0, len(image)])
#         # image[image >= 0] = 0
#         # image[image > 10] = 0
#         x, y = image.nonzero()
#         c = image[x, y]
#
#         im = plt.scatter(y[:], x[:], c=c[:], cmap='jet')
#         if n == 8:
#             plt.ylabel('Vertical Grid')
#         if n == 21:
#             plt.xlabel('Horizontal Grid')
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     # plt.show()
#     plt.savefig('old_models_error_by_avg.png')
#
# #read MAE and RMSE files
# readData = pd.read_csv('MAE.csv', header=None)
#
# #reading netcdf
# netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
# rain_models = netcdf_entire_dataset.variables['summing_models']
#
# imagesArr = []
# for i in range(2, 26):
#     minmaxArray = []
#     for y in range(46):  # 46 y-coordinates
#         for x in range(67):  # 67 x-coordinates
#             # for i in range(1, 25):  # for every model
#             print(y, x)
#             temp = []
#             data = np.array(rain_models[:20, :10, i-1, y, x])
#             minmaxArray.append(np.nanmean(data))
#
#     temp = np.array(readData[i])[1:]
#     temp[temp == ' '] = '0'
#     temp = pd.to_numeric(temp).reshape((46, 67))
#     minmaxArray = np.array(minmaxArray).reshape((46, 67))
#     f_temp = temp / minmaxArray
#     imagesArr.append(f_temp)
#
# show_images(imagesArr, 1, titles="Mean Absolute Error of Different Prediction Models divided by average rainfall")