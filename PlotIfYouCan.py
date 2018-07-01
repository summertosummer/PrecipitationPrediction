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


#2D Visualizaiton
def data_visualization_2dr(w_data, title, i=0, visualize=True):
    if visualize:
        plt.axis([0, len(w_data[0]), 0, len(w_data)])
        w_data[w_data <= 0] = 0
        w_data[w_data >= 100] = 0
        x, y = w_data.nonzero()
        # x = range(0, 65)
        # y = range(0, 44)
        c = w_data[x, y]
        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        plt.title(title)
        plt.colorbar()
        # plt.savefig('com/fig' + str(i) + '.png')
        plt.clim(0, 25)
        plt.show()
        plt.close()

def display_image(F, **kwargs):
    plt.figure()
    # F[F > -1000] = 0
    plt.imshow(F, **kwargs, origin='lower', cmap='jet')
    plt.clim(-500, 500)
    plt.colorbar()
    plt.show()

#read MAE and RMSE files
readData = pd.read_csv('25x25/MAE_25x25_Comparison.csv', header=None)

rmse = pd.to_numeric(np.array(readData[27])[1:]).reshape((44, 65))
print(rmse)
data_visualization_2dr(w_data=rmse, title='nothing')
# display_image(rmse)
