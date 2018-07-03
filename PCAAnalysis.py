from netCDF4 import Dataset
import numpy as np
import csv
import Orange as og
import itertools as it
from Orange.data import Domain, Table
from Orange.projection import PCA
import pandas as pd
import os

if not os.path.exists('pca'): os.makedirs('pca')

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

with open('random70.csv') as csvf:
    ind70 = csv.reader(csvf)
    indexi70 = list(ind70)
    index70 = indexi70[0]

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

def run_models(grid_y, grid_x):
    X_train, Y_train = create_training_data(grid_x, grid_y) # X and Y is the inputs and target
    data = Table(X_train, Y_train) # creating a Orange table combining both X and Y

    feature_method = og.preprocess.score.UnivariateLinearRegression() # feature selection
    selector = og.preprocess.SelectBestFeatures(method=feature_method, k=50) # taking 50 features out of 216
    out_data2 = selector(data) # this is the new dataset with 50 features

    pca = PCA(n_components=5) # PCA with 5 components
    model = pca(out_data2)
    train = model(out_data2)

    temp = []
    temp.append(pca.domain)
    for arr in model.components_:
        temp.append(list(arr))
    # temp.append(model.components_)
    np.savetxt('pca/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(temp), delimiter=',', fmt='%s')

def create_array(grid_y, grid_x):
    readData = pd.read_csv('pca/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
    arr = np.array(readData[:][:])
    arr = arr.astype(str)
    arr = np.char.replace(arr, " ", "")
    arr = np.char.replace(arr, "[", "")
    arr = np.char.replace(arr, "]", "")
    arr = np.char.replace(arr, "|Target", "")
    # print(arr[0][:])

    temp = [0]*24
    for f in range(0, 50):
        if arr[0][f] >= 'Feature001' and arr[0][f] <'Feature010':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[0] = temp[0] + value

        if arr[0][f] >= 'Feature010' and arr[0][f] <'Feature019':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[1] = temp[1] + value

        if arr[0][f] >= 'Feature019' and arr[0][f] <'Feature028':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[2] = temp[2] + value

        if arr[0][f] >= 'Feature028' and arr[0][f] <'Feature037':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[3] = temp[3] + value

        if arr[0][f] >= 'Feature037' and arr[0][f] <'Feature046':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[4] = temp[4] + value

        if arr[0][f] >= 'Feature046' and arr[0][f] <'Feature055':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[5] = temp[5] + value

        if arr[0][f] >= 'Feature055' and arr[0][f] <'Feature064':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[6] = temp[6] + value

        if arr[0][f] >= 'Feature064' and arr[0][f] <'Feature073':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[7] = temp[7] + value

        if arr[0][f] >= 'Feature073' and arr[0][f] <'Feature082':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[8] = temp[8] + value

        if arr[0][f] >= 'Feature082' and arr[0][f] <'Feature091':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[9] = temp[9] + value

        if arr[0][f] >= 'Feature091' and arr[0][f] <'Feature100':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[10] = temp[10] + value

        if arr[0][f] >= 'Feature100' and arr[0][f] <'Feature109':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[11] = temp[11] + value

        if arr[0][f] >= 'Feature109' and arr[0][f] <'Feature118':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[12] = temp[12] + value

        if arr[0][f] >= 'Feature118' and arr[0][f] <'Feature127':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[13] = temp[13] + value

        if arr[0][f] >= 'Feature127' and arr[0][f] <'Feature136':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[14] = temp[14] + value

        if arr[0][f] >= 'Feature136' and arr[0][f] <'Feature145':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[15] = temp[15] + value

        if arr[0][f] >= 'Feature145' and arr[0][f] <'Feature154':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[16] = temp[16] + value

        if arr[0][f] >= 'Feature154' and arr[0][f] <'Feature163':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[17] = temp[17] + value

        if arr[0][f] >= 'Feature163' and arr[0][f] <'Feature172':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[18] = temp[18] + value

        if arr[0][f] >= 'Feature172' and arr[0][f] <'Feature181':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[19] = temp[19] + value

        if arr[0][f] >= 'Feature181' and arr[0][f] <'Feature190':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[20] = temp[20] + value

        if arr[0][f] >= 'Feature190' and arr[0][f] <'Feature199':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[21] = temp[21] + value

        if arr[0][f] >= 'Feature199' and arr[0][f] <'Feature208':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[22] = temp[22] + value

        if arr[0][f] >= 'Feature208' and arr[0][f] <'Feature217':
            print(arr[0][f])
            value = float(arr[1][f]) + float(arr[2][f]) + float(arr[3][f]) + float(arr[4][f]) + float(arr[5][f])
            temp[23] = temp[23] + value

    return temp

finalArr = []
for grid_y in range(1, 45): # for every y
    for grid_x in range(1, 66): # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        run_models(grid_y, grid_x)
        arr = create_array(grid_y, grid_x)
        finalArr.append(arr)

np.savetxt('pca_analysis.csv', finalArr, delimiter=',', fmt='%10.5f')