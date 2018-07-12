from netCDF4 import Dataset
import numpy as np
import pandas as pd


def getArray1():
    arrCSV = pd.read_csv('array1.csv', header=None)
    array1 = []
    for i in range(24):
        temp = np.array(arrCSV[:])[i].reshape((44, 65))
        array1.append(temp)
    return np.array(array1)


def getArray2():
    arrCSV = pd.read_csv('array2.csv', header=None)
    array2 = []
    for i in range(24):
        temp = np.array(arrCSV[:])[i].reshape((44, 65))
        array2.append(temp)
    return np.array(array2)

print(getArray1())
print(getArray2())