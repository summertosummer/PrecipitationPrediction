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
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()

def display_image(F, **kwargs):
    plt.figure()
    # F[F > -1000] = 0
    plt.imshow(F, **kwargs, origin='lower', cmap='jet')
    plt.clim(-500, 500)
    plt.colorbar()
    plt.show()

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
        image[image >= 0] = 0
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
    plt.savefig('pca_analysis_imgN2.png')

def show_images3(images, cols, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
    # fig.suptitle('Plotting Precipitation Values. Day: 2016/05/20')
    count = 1
    for n, (image, title) in enumerate(zip(images, titles)):
        image1 = deepcopy(image)
        image2 = deepcopy(image)
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(8, 6, count)
        plt.axis([0, len(image1[0]), 0, len(image1)])
        image1[image1 < 0] = 0
        # image[image > 10] = 0
        x, y = image1.nonzero()
        c = image1[x, y]

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=0.5)
        # if n == 10:
        #     plt.ylabel('Vertical Grid')
        # if n == 22:
        #     plt.xlabel('Horizontal Grid')
        plt.colorbar()

        count += 1
        a2 = fig.add_subplot(8, 6, count)
        plt.axis([0, len(image2[0]), 0, len(image2)])
        image2[image2 >= 0] = 0
        # image[image > 10] = 0
        x2, y2 = image2.nonzero()
        c2 = image2[x2, y2]

        im = plt.scatter(y2[:], x2[:], c=c2[:], cmap='jet', s=0.5)
        # if n == 10:
        #     plt.ylabel('Vertical Grid')
        # if n == 22:
        #     plt.xlabel('Horizontal Grid')
        plt.colorbar()
        count +=1

    # plt.show()
    plt.savefig('pca_analysis_img7.png')

def show_images2(images):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=5, ncols=5)
    for image in images:
        plt.axis([0, len(image[0]), 0, len(image)])
        image[image < 5] = 0
        image[image > 10] = 0
        x, y = image.nonzero()
        c = image[x, y]
        # print(x.shape)
        # print(y.shape)
        # print(z.shape)
        # print(c.shape)

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet')

    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()

# #read MAE and RMSE files
# readData = pd.read_csv('pca_analysis.csv', header=None)
#
# imagesArr = []
# for i in range(24):
#     temp = pd.to_numeric(np.array(readData[i])[:]).reshape((44, 65))
#     imagesArr.append(temp)
#
# imagesArr = np.round(imagesArr/np.max(imagesArr), 1)
#
# show_images(imagesArr, 1)
# # data_visualization_2dr(w_data=imagesArr[0], title='model')
# # display_image(rmse)


#read MAE and RMSE files
readData = pd.read_csv('pca_diff.csv', header=None)
temp = pd.to_numeric(np.array(readData[4])[:]).reshape((44, 65))
data_visualization_2dr(temp, title='PCA5')
# display_image(rmse)
