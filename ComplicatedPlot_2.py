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

maxVal = 0

#read MAE and RMSE files
ifLR = pd.read_csv('new_results/reshapingIfLR.csv', header=None)
checkIfLR = np.array(ifLR[0])[:].reshape((44, 65))
checkIfLR = checkIfLR.astype(str)
checkIfLR = np.char.replace(checkIfLR, " ", "")

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

# read MAE and RMSE files
readDataMAE = pd.read_csv('new_results/MAE25x25_calculations_modified.csv', header=None)
tempMAE = pd.to_numeric(np.array(readDataMAE[33])[1:])
WAItself = pd.to_numeric(np.array(readDataMAE[29])[1:])
BestNew = pd.to_numeric(np.array(readDataMAE[31])[1:])

def create_array(grid_y, grid_x, f_ind):
    readData = pd.read_csv('pca2/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
    arr = np.array(readData[:][:])
    arr = arr.astype(str)
    arr = np.char.replace(arr, " ", "")
    arr = np.char.replace(arr, "[", "")
    arr = np.char.replace(arr, "]", "")
    arr = np.char.replace(arr, "|Target", "")

    readFeatures = pd.read_csv('unlucky13/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
    f_temp = np.array(readFeatures[0])[:]

    temp = [0]*24

    for num, f in enumerate(f_temp):
        p = 0
        f = np.char.replace(f, " ", "")
        print('there we go:', f)
        if f >= 'Feature001' and f < 'Feature010':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[2])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[0] += p # change index every time
            else:
                temp[0] = maxVal

        elif f >= 'Feature010' and f <'Feature019':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[3])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[1] += p  # change index every time
            else:
                temp[1] = maxVal

        elif f >= 'Feature019' and f <'Feature028':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[4])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[2] += p  # change index every time
            else:
                temp[2] = maxVal

        elif f >= 'Feature028' and f <'Feature037':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[5])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[3] += p  # change index every time
            else:
                temp[3] = maxVal

        elif f >= 'Feature037' and f <'Feature046':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[6])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[4] += p  # change index every time
            else:
                temp[4] = maxVal

        elif f >= 'Feature046' and f <'Feature055':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[7])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[5] += p  # change index every time
            else:
                temp[5] = maxVal

        elif f >= 'Feature055' and f <'Feature064':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[8])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[6] += p  # change index every time
            else:
                temp[6] = maxVal

        elif f >= 'Feature064' and f <'Feature073':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[9])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[7] += p  # change index every time
            else:
                temp[7] = maxVal

        elif f >= 'Feature073' and f <'Feature082':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[10])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[8] += p  # change index every time
            else:
                temp[8] = maxVal

        elif f >= 'Feature082' and f <'Feature091':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[11])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[9] += p  # change index every time
            else:
                temp[9] = maxVal

        elif f >= 'Feature091' and f <'Feature100':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[12])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[10] += p  # change index every time
            else:
                temp[10] = maxVal

        elif f >= 'Feature100' and f <'Feature109':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[13])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[11] += p  # change index every time
            else:
                temp[11] = maxVal

        elif f >= 'Feature109' and f <'Feature118':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[14])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[12] += p  # change index every time
            else:
                temp[12] = maxVal

        elif f >= 'Feature118' and f <'Feature127':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[15])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[13] += p  # change index every time
            else:
                temp[13] = maxVal

        elif f >= 'Feature127' and f <'Feature136':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[16])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[14] += p  # change index every time
            else:
                temp[14] = maxVal

        elif f >= 'Feature136' and f <'Feature145':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[17])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[15] += p  # change index every time
            else:
                temp[15] = maxVal

        elif f >= 'Feature145' and f <'Feature154':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[18])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[16] += p  # change index every time
            else:
                temp[16] = maxVal

        elif f >= 'Feature154' and f <'Feature163':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[19])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[17] += p  # change index every time
            else:
                temp[17] = maxVal

        elif f >= 'Feature163' and f <'Feature172':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[20])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[18] += p  # change index every time
            else:
                temp[18] = maxVal

        elif f >= 'Feature172' and f <'Feature181':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[21])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[19] += p  # change index every time
            else:
                temp[19] = maxVal

        elif f >= 'Feature181' and f <'Feature190':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[22])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[20] += p  # change index every time
            else:
                temp[20] = maxVal

        elif f >= 'Feature190' and f <'Feature199':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[23])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[21] += p  # change index every time
            else:
                temp[21] = maxVal

        elif f >= 'Feature199' and f <'Feature208':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[24])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[22] += p  # change index every time
            else:
                temp[22] = maxVal

        elif f >= 'Feature208' and f <'Feature217':
            print(f)
            readWeight = pd.to_numeric(np.array(readDataMAE[25])[1:])  # change index every time
            if BestNew[f_ind] >= readWeight[f_ind]:
                for f_pca in range(0, 50):
                    if arr[0][f_pca] == f:
                        p = float(arr[1][f_pca]) + float(arr[2][f_pca]) + float(arr[3][f_pca]) + float(
                            arr[4][f_pca]) + float(arr[5][f_pca])
                if checkIfLR[grid_y-1, grid_x-1] == 'LR':
                    tempRead = pd.read_csv('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', header=None)
                    tempo = pd.to_numeric(np.array(tempRead[0])[:])
                    p = p * tempo[num]
                elif WAItself[f_ind] == BestNew[f_ind]:
                    p = p * readWeight[f_ind]
                temp[23] += p  # change index every time
            else:
                temp[23] = maxVal

    return temp

# f_array = []
# f_index = 0
# for grid_y in range(1, 45): # for every y
#     for grid_x in range(1, 66): # for every x
#         print('=================PLACE:', grid_x, grid_y, '=====================')
#         tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
#         if not tempCheck.any():
#             f_array.append([0]*24)
#         else:
#             getArr = create_array(grid_y, grid_x, f_index)
#             f_array.append(getArr)
#             f_index += 1
#             # print(f_array)
# np.savetxt('complicated_2v3.csv', f_array, delimiter=',', fmt='%s')


def show_images(images, cols, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
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
        # plt.colorbar()
    # plt.show()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig('complicated_2v7.png')

#read MAE and RMSE files
readData = pd.read_csv('complicated_2v2.csv', header=None)
imagesArr = []
for i in range(24):
    temp = pd.to_numeric(np.array(readData[i])[:]).reshape((44, 65))
    temp[temp<0] = 0
    temp[temp>50] = 50
    imagesArr.append(temp)

# imagesArr = np.round(imagesArr/np.max(imagesArr), 1)
show_images(imagesArr, 1)