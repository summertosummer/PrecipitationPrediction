import numpy as np
from netCDF4 import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt

Threshold = 2

def firstPass(inputImg):
    global labeling
    labeling = np.zeros(shape=(len(inputImg), len(inputImg[0])))
    global eLabels
    eLabels = defaultdict(list)
    global newLabel
    newLabel = 1
    flag = False
    # first pass
    for i in range(len(inputImg)):
        for j in range(len(inputImg[0])):
            minLabel = newLabel

            if j - 1 >= 0 and same_component_check(inputImg[i][j], inputImg[i][j - 1]):
                if minLabel > labeling[i][j - 1] and labeling[i][j - 1] != 0:
                    minLabel = labeling[i][j - 1]
                    flag = True

            if i - 1 >= 0 and same_component_check(inputImg[i][ j], inputImg[i - 1][ j]):
                if minLabel > labeling[i - 1][j] and labeling[i - 1][j] != 0:
                    minLabel = labeling[i - 1][j]
                    flag = True

            if i - 1 >= 0 and j - 1 >= 0 and same_component_check(inputImg[i][j], inputImg[i - 1][ j - 1]):
                if minLabel > labeling[i - 1][j - 1] and labeling[i - 1][j - 1] != 0:
                    minLabel = labeling[i - 1][j - 1]
                    flag = True

            if i - 1 >= 0 and j + 1 < len(inputImg[0]) and same_component_check(inputImg[i][j], inputImg[i - 1][ j + 1]):
                if minLabel > labeling[i - 1][j + 1] and labeling[i - 1][j + 1] != 0:
                    minLabel = labeling[i - 1][j + 1]
                    flag = True

            if flag == False: newLabel += 1

            flag = False
            labeling[i, j] = minLabel

            # tracking the minimum labeling
            if j - 1 >= 0 and same_component_check(inputImg[i][j], inputImg[i][j - 1]):
                if not isExist(labeling[i][j], labeling[i][j-1]):
                    eLabels[int(labeling[i][j])].append(labeling[i][j - 1])
                if not isExist(labeling[i][j-1], labeling[i][j]):
                    eLabels[int(labeling[i][j - 1])].append(labeling[i][j])

            if i - 1 >= 0 and same_component_check(inputImg[i][j], inputImg[i - 1][j]):
                if not isExist(labeling[i][j], labeling[i-1][j]):
                    eLabels[int(labeling[i][j])].append(labeling[i-1][j])
                if not isExist(labeling[i-1][j], labeling[i][j]):
                    eLabels[int(labeling[i-1][j])].append(labeling[i][j])

            if i - 1 >= 0 and j - 1 >= 0 and same_component_check(inputImg[i][j], inputImg[i - 1][j - 1]):
                if not isExist(labeling[i][j], labeling[i-1][j-1]):
                    eLabels[int(labeling[i][j])].append(labeling[i-1][j-1])
                if not isExist(labeling[i-1][j-1], labeling[i][j]):
                    eLabels[int(labeling[i-1][j-1])].append(labeling[i][j])

            if i - 1 >= 0 and j + 1 < len(inputImg[0]) and same_component_check(inputImg[i][j], inputImg[i - 1][j + 1]):
                if not isExist(labeling[i][j], labeling[i-1][j+1]):
                    eLabels[int(labeling[i][j])].append(labeling[i-1][j+1])
                if not isExist(labeling[i-1][j+1], labeling[i][j]):
                    eLabels[int(labeling[i-1][j+1])].append(labeling[i][j])

def secondPass(inputImg):
    for i in range(len(inputImg)):
        for j in range(len(inputImg[0])):
            if labeling[i][j] != 0:
                labeling[i][j] = findMinimum(labeling[i][j])

def findMinimum(x):
    min = int(x)
    for y in eLabels[int(x)]:
        if y < min and y != 0:
            min = y
    return min

def isExist(x, i0):
    mark = False
    for a in eLabels[int(x)]:
        if a== int(i0):
            mark = True
            break
        else:
            mark = False
    return mark

def same_component_check(point1, point2):
    return abs(point1 - point2) <= Threshold

def output(inputImg):
    for value in range(1, newLabel):
        flag = False
        print('--------------------',value)
        arr = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
        arr[:] = ' '
        for j in range(len(labeling)):
            for i in range(len(labeling[0])):
                if (labeling[j][i] == value):
                    flag = True
                    arr[j][i] = str(inputImg[j][i])
        if flag: print(arr)
        # data_visualization_2d(np.array(arr), 'Real Data', visualize=True)

if __name__ == "__main__":
    input1 = [[255, 257, 55, 0],
             [253, 0, 15, 0],
             [251, 252, 252, 0],
             [249, 0, 22, 0],
             [255, 25, 255, 0],
             [0, 0, 0, 0],
             [110, 23, 255, 0],
             [0, 0, 0, 0],
             [99, 99, 99, 0]]

    firstPass(input1)
    secondPass(input1)
    print(labeling)

    output(input1)