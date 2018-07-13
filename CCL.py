import numpy as np
from netCDF4 import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt

Threshold = 2
Background = 5

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

#2D Visualizaiton
def data_visualization_2dr(w_data, title, i=0, visualize=True):
    if visualize:
        w_data[w_data == ' '] = np.nan # making all the NONE value NAN
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
        plt.gca().invert_yaxis()
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()

def output1(inputImg):
    finalArr1 = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
    finalArr1[:] = ' '
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

        if flag and (arr != ' ').sum() > Background:
            print(arr)
            for j in range(len(labeling)):
                for i in range(len(labeling[0])):
                    if (labeling[j][i] == value):
                        finalArr1[j][i] = str(arr[j][i])

    data_visualization_2dr(np.array(finalArr1), 'Connected Components', visualize=True)

def output2(inputImg):
    finalArr1 = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
    finalArr1[:] = ' '
    for value in range(1, newLabel):
        flag = False
        print('--------------------',value)
        arr = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
        arr[:] = ' '
        for j in range(len(labeling)):
            for i in range(len(labeling[0])):
                if (labeling[j][i] == value):
                    flag = True
                    arr[j][i] = str(value)

        if flag and (arr != ' ').sum() > Background:
            print(arr)
            for j in range(len(labeling)):
                for i in range(len(labeling[0])):
                    if (labeling[j][i] == value):
                        finalArr1[j][i] = str(arr[j][i])

    data_visualization_2dr(np.array(finalArr1), 'Connected Components', visualize=True)

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

    output1(input1)
    output2(input1)



# def output(inputImg):
#     finalArr1 = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
#     finalArr1[:] = ' '
#     for value in range(1, newLabel):
#         flag = False
#         print('--------------------',value)
#         arr = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
#         arr[:] = ' '
#         for j in range(len(labeling)):
#             for i in range(len(labeling[0])):
#                 if (labeling[j][i] == value):
#                     flag = True
#                     arr[j][i] = str(inputImg[j][i])
#         if flag and (arr != ' ').sum() > Background:
#             print(arr)
#             finalArr1[finalArr1 == ' '] = np.nan
#             arr[arr == ' '] = np.nan
#             # np.array(list(map(int, finalArr1))), np.array(list(map(int, arr)))
#             finalArr1 = np.nansum(np.dstack((finalArr1.astype(np.float), arr.astype(np.float))), 2)
#             finalArr1 = finalArr1.astype(np.object)
#             print(finalArr1)
#             finalArr1[finalArr1 == np.nan] = ' '
#
#     data_visualization_2dr(np.array(finalArr1), 'Connected Components', visualize=True)