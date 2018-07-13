'''
Define a constant - threshold
Set this to 10%

The arrays have  the following dimensions: M *x*y

Array1:
For each model m
     For each grid point xy
           If   [error of m - error of best model ( new or old) ]/ error of best model
                Is less than threshold
                    Then array1[m][x][y]=1 else 0



Array 2:
For each model m
     For each grid point xy
           If   [error of m - error of best old model ]/ error of best model
                Is less than threshold
                    Then array1[m][x][y]=1 else 0

The difference b twenty the two arrays is the the first one considers all around best model
while the second one considers only old models
'''

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from copy import deepcopy
np.seterr(divide='ignore', invalid='ignore')

threshold = 0.10

# read MAE and RMSE files
readDataMAE = pd.read_csv('new_results/RMSE25x25_calculations_modified.csv', header=None)
AllBest = pd.to_numeric(np.array(readDataMAE[34])[1:])
OldBest = pd.to_numeric(np.array(readDataMAE[26])[1:])

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

def array1():
    first_array = []
    for i in range(2, 26):
        print('xxxxxxxxxxxxxxxxxxxxxxx', i - 1, 'xxxxxxxxxxxxxxxxxxxxxxxxxx')
        readError = pd.to_numeric(np.array(readDataMAE[i])[1:])  # change index every time
        temp = (readError - AllBest) / AllBest
        # print((temp < .10).sum())
        temp2 = deepcopy(temp)
        temp2[temp < threshold] = 1
        temp2[temp >= threshold] = 0

        # resizing, it takes time
        f_array = []
        f_index = 0
        for grid_y in range(1, 45):  # for every y
            for grid_x in range(1, 66):  # for every x
                print('=================PLACE:', grid_x, grid_y, '=====================')
                tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
                if not tempCheck.any():
                    f_array.append(' ')
                else:
                    f_array.append(str(temp2[f_index]))
                    f_index += 1

        first_array.append(f_array)
    return np.array(first_array)

def array2():
    second_array = []
    for i in range(2, 26):
        print('xxxxxxxxxxxxxxxxxxxxxxx', i - 1, '############################')
        readError = pd.to_numeric(np.array(readDataMAE[i])[1:])  # change index every time
        temp = (readError - OldBest) / OldBest
        temp2 = deepcopy(temp)
        temp2[temp < threshold] = 1
        temp2[temp >= threshold] = 0

        # resizing, it takes time
        f_array = []
        f_index = 0
        for grid_y in range(1, 45):  # for every y
            for grid_x in range(1, 66):  # for every x
                print('=================PLACE:', grid_x, grid_y, '=====================')
                tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
                if not tempCheck.any():
                    f_array.append(' ')
                else:
                    f_array.append(str(temp2[f_index]))
                    f_index += 1

        second_array.append(f_array)
    return np.array(second_array)

# arr1 = array1()
arr2 = array2()
# np.savetxt('array1.csv', arr1, delimiter=',', fmt='%s')
np.savetxt('array2.csv', arr2, delimiter=',', fmt='%s')
# print(arr1)
print(arr2)