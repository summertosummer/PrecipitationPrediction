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

# total = 0
# countMAE = 0
# countRMSE = 0
# tempCheck = []
# for grid_y in range(1, 45): # for every y
#     for grid_x in range(1, 66): # for every x
#         print('=================PLACE:', grid_x, grid_y, '=====================')
#         tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
#         if not tempCheck.any():
#             continue
#         run_models(grid_y, grid_x)

###########################################################################################
# #read MAE and RMSE files
# readData = pd.read_csv('new_results/MAE25x25_calculations_modified.csv', header=None)
# temp = pd.to_numeric(np.array(readData[33])[1:])
# print(temp)
#
# f_array = []
# f_index = 0
# for grid_y in range(1, 45): # for every y
#     for grid_x in range(1, 66): # for every x
#         print('=================PLACE:', grid_x, grid_y, '=====================')
#         tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
#         if not tempCheck.any():
#             f_array.append(0)
#         else:
#             readFeatures = pd.read_csv('unlucky13/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
#             f_temp = np.array(readFeatures[0])[:]
#
#             flag = False
#             for f in f_temp:
#                 if f == 'Feature 058' and temp[f_index] > 0:
#                     print(f)
#                     flag = True
#
#             if flag: f_array.append(1)
#             else: f_array.append(0)
#             f_index += 1
#             # print(f_array)
# np.savetxt('complicated_1.csv', f_array, delimiter=',', fmt='%s')
###########################################################################################

def create_array(grid_y, grid_x, tempMAE):
    readFeatures = pd.read_csv('unlucky13/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
    f_temp = np.array(readFeatures[0])[:]

    temp = [0]*24
    for f in f_temp:
        f = np.char.replace(f, " ", "")
        print('there we go:', f)
        if f >= 'Feature001' and f <'Feature010' and tempMAE > 0:
            print(f)
            temp[0] += 1

        elif f >= 'Feature010' and f <'Feature019' and tempMAE > 0:
            print(f)
            temp[1] += 1

        elif f >= 'Feature019' and f <'Feature028' and tempMAE > 0:
            print(f)
            temp[2] += 1

        elif f >= 'Feature028' and f <'Feature037' and tempMAE > 0:
            print(f)
            temp[3] += 1

        elif f >= 'Feature037' and f <'Feature046' and tempMAE > 0:
            print(f)
            temp[4] += 1

        elif f >= 'Feature046' and f <'Feature055' and tempMAE > 0:
            print(f)
            temp[5] += 1

        elif f >= 'Feature055' and f <'Feature064' and tempMAE > 0:
            print(f)
            temp[6] += 1

        elif f >= 'Feature064' and f <'Feature073' and tempMAE > 0:
            print(f)
            temp[7] += 1

        elif f >= 'Feature073' and f <'Feature082' and tempMAE > 0:
            print(f)
            temp[8] += 1

        elif f >= 'Feature082' and f <'Feature091' and tempMAE > 0:
            print(f)
            temp[9] += 1

        elif f >= 'Feature091' and f <'Feature100' and tempMAE > 0:
            print(f)
            temp[10] += 1

        elif f >= 'Feature100' and f <'Feature109' and tempMAE > 0:
            print(f)
            temp[11] += 1

        elif f >= 'Feature109' and f <'Feature118' and tempMAE > 0:
            print(f)
            temp[12] += 1

        elif f >= 'Feature118' and f <'Feature127' and tempMAE > 0:
            print(f)
            temp[13] += 1

        elif f >= 'Feature127' and f <'Feature136' and tempMAE > 0:
            print(f)
            temp[14] += 1

        elif f >= 'Feature136' and f <'Feature145' and tempMAE > 0:
            print(f)
            temp[15] += 1

        elif f >= 'Feature145' and f <'Feature154' and tempMAE > 0:
            print(f)
            temp[16] += 1

        elif f >= 'Feature154' and f <'Feature163' and tempMAE > 0:
            print(f)
            temp[17] += 1

        elif f >= 'Feature163' and f <'Feature172' and tempMAE > 0:
            print(f)
            temp[18] += 1

        elif f >= 'Feature172' and f <'Feature181' and tempMAE > 0:
            print(f)
            temp[19] += 1

        elif f >= 'Feature181' and f <'Feature190' and tempMAE > 0:
            print(f)
            temp[20] += 1

        elif f >= 'Feature190' and f <'Feature199' and tempMAE > 0:
            print(f)
            temp[21] += 1

        elif f >= 'Feature199' and f <'Feature208' and tempMAE > 0:
            print(f)
            temp[22] += 1

        elif f >= 'Feature208' and f <'Feature217' and tempMAE > 0:
            print(f)
            temp[23] += 1

    return temp



# # read MAE and RMSE files
# readDataMAE = pd.read_csv('new_results/MAE25x25_calculations_modified.csv', header=None)
# tempMAE = pd.to_numeric(np.array(readDataMAE[33])[1:])
#
# f_array = []
# f_index = 0
# for grid_y in range(1, 45): # for every y
#     for grid_x in range(1, 66): # for every x
#         print('=================PLACE:', grid_x, grid_y, '=====================')
#         tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
#         if not tempCheck.any():
#             f_array.append([0]*24)
#         else:
#             getArr = create_array(grid_y, grid_x, tempMAE[f_index])
#             f_array.append(getArr)
#             f_index += 1
#             # print(f_array)
# np.savetxt('complicated_1.csv', f_array, delimiter=',', fmt='%s')


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
        # plt.colorbar()
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()

def show_images(images, cols, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 12), dpi=300, facecolor='w', edgecolor='k')
    # fig.suptitle('Plotting Precipitation Values. Day: 2016/05/20')
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(6, 4, n + 1)
        plt.axis([0, len(image[0]), 0, len(image)])
        # image[image >= 0] = 0
        # image[image > 10] = 0
        x, y = image.nonzero()
        c = image[x, y]

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=1)
        # if n == 10:
        #     plt.ylabel('Vertical Grid')
        # if n == 22:
        #     plt.xlabel('Horizontal Grid')
        plt.colorbar()
    # plt.show()
    plt.savefig('complicated_1.png')

#read MAE and RMSE files
readData = pd.read_csv('complicated_1.csv', header=None)
imagesArr = []
for i in range(24):
    temp = pd.to_numeric(np.array(readData[i])[:]).reshape((44, 65))
    imagesArr.append(temp)

# imagesArr = np.round(imagesArr/np.max(imagesArr), 1)
show_images(imagesArr, 1)

# #read MAE and RMSE files
# readData = pd.read_csv('complicated_1.csv', header=None)
# temp = pd.to_numeric(np.array(readData[0])[:]).reshape((44, 65))
# data_visualization_2dr(temp, title='Complicated Plot 1')