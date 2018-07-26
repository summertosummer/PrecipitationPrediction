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
readData = pd.read_csv('final_results/ModelsInfo25x25_modified_final_calculation.csv', header=None)
# read MAE and RMSE files
readData2 = pd.read_csv('new_results/RMSE25x25_calculations_modified.csv', header=None)

AllBest = pd.to_numeric(np.array(readData[24])[1:2391])
# print(temp)

# def data_visualization_2dr(w_data, title, i=0, visualize=True):
#     if visualize:
#         min_v = np.nanmin(w_data)
#         max_v = np.nanmax(w_data[w_data != np.inf])
#         print(min_v, max_v)
#         plt.axis([0, len(w_data[0]), 0, len(w_data)])
#         # plt.axis([32, len(w_data[0]), 22, len(w_data)])
#         # w_data[w_data < 0] = 0
#         # w_data[w_data >= 100] = 0
#         x, y = w_data.nonzero()
#         # x = range(0, 65)
#         # y = range(0, 44)
#         c = w_data[x, y]
#         plt.scatter(y[:], x[:], c=c[:], cmap='jet') #cubehelix
#         plt.title(title)
#         plt.colorbar()
#         plt.ylabel('Vertical Grid')
#         plt.xlabel('Horizontal Grid')
#         # plt.savefig('com/fig' + str(i) + '.png')
#         # plt.clim(min_v, max_v)
#         plt.show()
#         plt.close()

def data_visualization_2dr(w_data, title, i=0, visualize=True):
    if visualize:
        min_v = np.nanmin(w_data)
        max_v = np.nanmax(w_data[w_data != np.inf])
        print(min_v, max_v)
        plt.axis([0, len(w_data[0]), 0, len(w_data)])
        # plt.axis([32, len(w_data[0]), 22, len(w_data)])
        # w_data[w_data < 0] = 0
        # w_data[w_data >= 100] = 0
        w_data1, w_data2 = deepcopy(w_data), deepcopy(w_data)
        w_data1[w_data1<0] = 0
        w_data2[w_data2>=0] = 0
        x1, y1 = w_data1.nonzero()
        c1 = w_data1[x1, y1]

        x2, y2 = w_data2.nonzero()
        c2 = w_data2[x2, y2]

        pa = plt.scatter(y1[:], x1[:], c=c1[:], cmap='cool', vmin=0, vmax=max_v) #cubehelix
        cba = plt.colorbar(pa)
        pb = plt.scatter(y2[:], x2[:], c=c2[:], cmap='autumn', vmin=min_v, vmax=-0.0001)  # cubehelix
        cbb = plt.colorbar(pb)
        cba.set_label('positive')
        cbb.set_label('negative')

        plt.title(title)
        plt.ylabel('Vertical Grid')
        plt.xlabel('Horizontal Grid')
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(min_v, max_v)
        plt.show()
        plt.close()

#
def show_images(images, cols, titles):
    min_v = np.nanmin(images)
    max_v = np.nanmax(images[images != np.inf])
    print(min_v, max_v)
    # assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 10), dpi=100, facecolor='w', edgecolor='k')
    fig.suptitle(titles)
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(2, 2, n + 1)
        plt.axis([0, len(image[0]), 0, len(image)])
        # image[image >= 0] = 0
        # image[image > 10] = 0
        w_data1, w_data2 = deepcopy(image), deepcopy(image)
        w_data1[w_data1 < 0] = 0
        w_data2[w_data2 >= 0] = 0
        x1, y1 = w_data1.nonzero()
        c1 = w_data1[x1, y1]

        x2, y2 = w_data2.nonzero()
        c2 = w_data2[x2, y2]

        pa = plt.scatter(y1[:], x1[:], c=c1[:], cmap='cool', vmin=0.0001, vmax=max_v, s=10)  # cubehelix

        # if n == 1: plt.colorbar(pa)

        if n == 3:
            pb = plt.scatter(y2[:], x2[:], c=c2[:], cmap='autumn', vmin=min_v, vmax=-0.0001, s=10)  # cubehelix
            # plt.colorbar(pb)
            # plt.colorbar(pa)
            # plt.colorbar(pb)

        # im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=2)
        # if n == 8:
        #     plt.ylabel('Vertical Grid')
        if n == 0:
            plt.title('(a)')
        if n == 1:
            plt.title('(b)')
        if n == 2:
            plt.title('(c)')
        if n == 3:
            plt.title('(d)')
        # plt.clim(min_v, max_v)
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    cbar_ax = fig.add_axes([0.92, 0.505, 0.01, 0.4])
    cba = fig.colorbar(pa, cax=cbar_ax)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.01, 0.4])
    cbb = fig.colorbar(pb, cax=cbar_ax)
    # plt.show()
    plt.savefig('analyze_new_model_error1.png')


def newBestAmongAllPlot():
    NewBest = pd.to_numeric(np.array(readData[21])[1:2391])
    Diff = pd.to_numeric(np.array(readData[23])[1:2391])
    NewBest[Diff < 0] = np.nan
    return NewBest

def bestAmongNewPlot():
    NewBest = pd.to_numeric(np.array(readData[21])[1:2391])
    return NewBest

def diffNewOldPlot():
    Diff = pd.to_numeric(np.array(readData[23])[1:2391])
    return Diff

def oldBestPlot():
    oldBest = pd.to_numeric(np.array(readData2[26])[1:])
    return oldBest


newBestAmongAll = newBestAmongAllPlot()
bestAmongNew = bestAmongNewPlot()
diffNewOld = diffNewOldPlot()
oldBest = oldBestPlot()

f_newBestAmongAll = []
f_bestAmongNew = []
f_diffNewOld = []
f_oldBest = []
f_index = 0
for grid_y in range(1, 45): # for every y
    for grid_x in range(1, 66): # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
        if not tempCheck.any():
            f_newBestAmongAll.append(0)
            f_bestAmongNew.append(0)
            f_diffNewOld.append(0)
            f_oldBest.append(0)
        else:
            f_newBestAmongAll.append(newBestAmongAll[f_index])
            f_bestAmongNew.append(bestAmongNew[f_index])
            f_diffNewOld.append(diffNewOld[f_index])
            f_oldBest.append(oldBest[f_index])
            f_index += 1

final = []
final.append(np.array(f_oldBest).reshape((44, 65)))
final.append(np.array(f_bestAmongNew).reshape((44, 65)))

final.append(np.array(f_newBestAmongAll).reshape((44, 65)))
final.append(np.array(f_diffNewOld).reshape((44, 65)))

show_images(np.array(final), 1, titles="Analyzing errors of new and old models")

# data_visualization_2dr(np.array(f_newBestAmongAll).reshape((44, 65)), title='Plotting the error of new model beats old models')
# data_visualization_2dr(np.array(f_bestAmongNew).reshape((44, 65)), title='Plotting the error of new best model')
# data_visualization_2dr(np.array(f_diffNewOld).reshape((44, 65)), title='Plotting the difference between best new and best old model')

# np.savetxt('new_results/n_reshapingIfLR.csv', f_array, delimiter=',', fmt='%s')
# print(np.array(f_array).reshape((44, 65)))


# #read MAE and RMSE files
# readData = pd.read_csv('new_results/reshaping25x25mae.csv', header=None)
# temp = pd.to_numeric(np.array(readData[0])[:]).reshape((44, 65))
# data_visualization_2dr(temp, title='MAE')





# import pickle
# import tensorflow as tf
# from netCDF4 import Dataset
# import pickle
# import numpy as np
# import csv
# import Orange as og
# import math
# import itertools as it
# from Orange.data import Domain, Table
# import random
# from Orange.projection import PCA
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import pandas as pd
# import os
# from copy import deepcopy
#
# #access netcdf data file
# netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
# rain_models = netcdf_entire_dataset.variables['summing_models']
#
# #read MAE and RMSE files
# readData = pd.read_csv('final_results/ModelsInfo25x25_modified_final_calculation.csv', header=None)
# temp = np.array(readData[2])[1:2391]
# print(temp)
#
# def data_visualization_2dr(w_data, title, i=0, visualize=True):
#     if visualize:
#         plt.axis([0, len(w_data[0]), 0, len(w_data)])
#         w_data[w_data < 0] = 0
#         # w_data[w_data >= 100] = 0
#         x, y = w_data.nonzero()
#         # x = range(0, 65)
#         # y = range(0, 44)
#         c = w_data[x, y]
#         plt.scatter(y[:], x[:], c=c[:], cmap='jet')
#         plt.title(title)
#         plt.colorbar()
#         # plt.savefig('com/fig' + str(i) + '.png')
#         # plt.clim(-5, 0)
#         plt.show()
#         plt.close()
# #
# f_array = []
# f_index = 0
# for grid_y in range(1, 45): # for every y
#     for grid_x in range(1, 66): # for every x
#         print('=================PLACE:', grid_x, grid_y, '=====================')
#         tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
#         if not tempCheck.any():
#             f_array.append(0)
#         else:
#             f_array.append(temp[f_index])
#             f_index += 1
#
# np.savetxt('new_results/n_reshapingIfLR.csv', f_array, delimiter=',', fmt='%s')
# print(np.array(f_array).reshape((44, 65)))
#
# # #read MAE and RMSE files
# # readData = pd.read_csv('new_results/reshaping25x25mae.csv', header=None)
# # temp = pd.to_numeric(np.array(readData[0])[:]).reshape((44, 65))
# # data_visualization_2dr(temp, title='MAE')
#
