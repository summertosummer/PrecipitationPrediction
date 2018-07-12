import numpy as np
from netCDF4 import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

#reading netcdf
# netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
# rain_models = netcdf_entire_dataset.variables['rain_models']
#
# inp1 = rain_models[19, 19, 12, 300:500, 1400:1600]
# inp2 = rain_models[19, 18, 9, 300:500, 1400:1600]
# input = [[255, 0, 55, 0],
#          [15, 0, 15, 0],
#          [255, 0, 255, 0],
#          [21, 0, 22, 0],
#          [255, 25, 255, 0],
#          [0, 0, 0, 0],
#          [110, 23, 255, 0],
#          [0, 0, 0, 0],
#          [321, 0, 99, 0]]
# input = [[255, 0, 0, 0],
#          [255, 0, 255, 0],
#          [255, 0, 255, 0],
#          [255, 0, 255, 0],
#          [255, 255, 255, 0],
#          [0, 0, 0, 0],
#          [0, 255, 255, 0],
#          [0, 255, 255, 0],
#          [255, 0, 0, 0]]

#read MAE and RMSE files
readData = pd.read_csv('25x25/MAE_25x25_Comparison.csv', header=None)

rmse = pd.to_numeric(np.array(readData[27])[1:]).reshape((44, 65))

def firstPass(input):
    global labeling
    labeling = np.zeros(shape=(len(input), len(input[0])))
    global eLabels
    eLabels = defaultdict(list)
    global newLabel
    newLabel = 1
    flag = False
    for i in range(len(input)):
        for j in range(len(input[0])):
            minLabel = newLabel

            if not aboveThreshold(input[i][j]):
                labeling[i][j] = 0
            else:
                if j-1>=0 and aboveThreshold(input[i][j-1]):
                    if minLabel > labeling[i][j-1] and labeling[i][j-1]!=0:
                        minLabel = labeling[i][j - 1]
                        flag = True

                if i-1>=0 and aboveThreshold(input[i-1][j]):
                    if minLabel > labeling[i-1][j] and labeling[i-1][j]!=0:
                        minLabel = labeling[i-1][j]
                        flag = True

                if i-1>=0 and j-1>=0 and aboveThreshold(input[i-1][j-1]):
                    if minLabel > labeling[i-1][j-1] and labeling[i-1][j-1]!=0:
                        minLabel = labeling[i-1][j-1]
                        flag = True

                if i-1>=0 and j+1<len(input[0]) and aboveThreshold(input[i-1][j+1]):
                    if minLabel > labeling[i-1][j+1] and labeling[i-1][j+1]!=0:
                        minLabel = labeling[i-1][j+1]
                        flag = True

                if flag == False:
                    newLabel = newLabel + 1

                flag = False
                labeling[i][j] = minLabel

            if j - 1 >= 0 and aboveThreshold(input[i][j]) and aboveThreshold(input[i][j - 1]):
                if not isExist(labeling[i][j], labeling[i][j-1]):
                    eLabels[int(labeling[i][j])].append(labeling[i][j - 1])
                if not isExist(labeling[i][j-1], labeling[i][j]):
                    eLabels[int(labeling[i][j - 1])].append(labeling[i][j])

            if i - 1 >= 0 and aboveThreshold(input[i][j]) and aboveThreshold(input[i - 1][j]):
                if not isExist(labeling[i][j], labeling[i-1][j]):
                    eLabels[int(labeling[i][j])].append(labeling[i-1][j])
                if not isExist(labeling[i-1][j], labeling[i][j]):
                    eLabels[int(labeling[i-1][j])].append(labeling[i][j])

            if i - 1 >= 0 and j - 1 >= 0 and aboveThreshold(input[i][j]) and aboveThreshold(input[i - 1][j - 1]):
                if not isExist(labeling[i][j], labeling[i-1][j-1]):
                    eLabels[int(labeling[i][j])].append(labeling[i-1][j-1])
                if not isExist(labeling[i-1][j-1], labeling[i][j]):
                    eLabels[int(labeling[i-1][j-1])].append(labeling[i][j])

            if i - 1 >= 0 and j + 1 < len(input[0]) and aboveThreshold(input[i][j]) and aboveThreshold(input[i - 1][j + 1]):
                if not isExist(labeling[i][j], labeling[i-1][j+1]):
                    eLabels[int(labeling[i][j])].append(labeling[i-1][j+1])
                if not isExist(labeling[i-1][j+1], labeling[i][j]):
                    eLabels[int(labeling[i-1][j+1])].append(labeling[i][j])
    # totalLabel = newLabel
    # print(newLabel)

def secondPass(input):
    for i in range(len(input)):
        for j in range(len(input[0])):
            if labeling[i][j] != 0:
                labeling[i][j] = findMinimum(labeling[i][j])

def contour_plot_2d(w_data, title, visualize=True):
    xlist, ylist = w_data.nonzero()
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X ** 2 + Y ** 2)
    plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

def data_visualization_2d(w_data, title, visualize=True):
    if visualize:
        # w_data[w_data<5] = 0
        x, y = w_data.nonzero()
        c = w_data[x, y]
        # print(x.shape)
        # print(y.shape)
        # print(z.shape)
        # print(c.shape)

        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        plt.title(title)
        plt.colorbar()
        plt.show()

xmin = 0
xmax = 200
ymin = 0
ymax = 200
def display_image(F, **kwargs):
    plt.figure()
    plt.imshow(F, **kwargs)
    plt.show()

def show_images(images, cols=20, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(9, 8, n + 1)
        x, y = image.nonzero()
        c = image[x, y]
        # print(x.shape)
        # print(y.shape)
        # print(z.shape)
        # print(c.shape)

        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        # a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.colorbar()
    plt.show()

def output(input):
    for value in range(1, newLabel):
        print(value)
        arr = np.zeros(shape=(len(input), len(input[0])))
        for j in range(len(labeling)):
            for i in range(len(labeling[0])):
                if (labeling[j][i] == value):
                    arr[j][i] = input[j][i]
        print(len(arr), len(arr[0]))
        data_visualization_2d(np.array(arr), 'Real Data', visualize=True)

def singleOutput(ccl, input):
    arr = np.zeros(shape=(len(input), len(input[0])))
    for j in range(len(labeling)):
        for i in range(len(labeling[0])):
            if (labeling[j][i] == ccl):
                arr[j][i] = input[j][i]
    # print(len(arr), len(arr[0]))
    # data_visualization_2d(np.array(arr), 'Real Data', visualize=True)
    display_image(np.array(arr), extent=[xmin, xmax, ymin, ymax], origin='lower')
    return arr


def isExist(x, i0):
    mark = False
    for a in eLabels[int(x)]:
        if a== int(i0):
            mark = True
            break
        else:
            mark = False
    return mark

def findMinimum(x):
    min = int(x)
    for y in eLabels[int(x)]:
        if y < min and y != 0:
            min = y
    return min

def aboveThreshold(now):
    trigger = False
    threshold = 15
    if int(now) >= threshold:
        trigger = True
    else:
        trigger = False
    return trigger

# q = inp1
# q[q<=5] = 0
# data_visualization_2d(np.array(q), 'Full Data', visualize=True)
# data_visualization_2d(np.array(input2), 'Full Data', visualize=True)
# display_image(np.array(q), extent=[xmin, xmax, ymin, ymax], origin='lower')
# firstPass()
# # output()
# # print(newLabel)
# secondPass()
# output()

firstPass(rmse)
secondPass(rmse)
output(rmse)

# firstPass(inp2)
# secondPass(inp2)
# singleOutput(2, rmse)




# import numpy as np
# from netCDF4 import Dataset
# from collections import defaultdict
# import matplotlib.pyplot as plt
#
# def connectedComponents(inputImg):
#     newLabel = 1
#     flag = False
#     global labeling
#     labeling = np.zeros(shape=(len(inputImg), len(inputImg[0])))
#
#     # first pass
#     for i in range(len(inputImg)):
#         for j in range(len(inputImg[0])):
#             minLabel = newLabel
#
#             if j - 1 >= 0 and inputImg[i][j] == inputImg[i][j - 1]:
#                 if minLabel > labeling[i][j - 1] and labeling[i][j - 1] != 0:
#                     minLabel = labeling[i][j - 1]
#                     flag = True
#
#             if i - 1 >= 0 and inputImg[i][ j] == inputImg[i - 1][ j]:
#                 if minLabel > labeling[i - 1][j] and labeling[i - 1][j] != 0:
#                     minLabel = labeling[i - 1][j]
#                     flag = True
#
#             if i - 1 >= 0 and j - 1 >= 0 and inputImg[i][j] == inputImg[i - 1][ j - 1]:
#                 if minLabel > labeling[i - 1][j - 1] and labeling[i - 1][j - 1] != 0:
#                     minLabel = labeling[i - 1][j - 1]
#                     flag = True
#
#             if i - 1 >= 0 and j + 1 < len(inputImg[0]) and inputImg[i][j] == inputImg[i - 1][ j + 1]:
#                 if minLabel > labeling[i - 1][j + 1] and labeling[i - 1][j + 1] != 0:
#                     minLabel = labeling[i - 1][j + 1]
#                     flag = True
#
#             if flag == False: newLabel += 1
#
#             flag = False
#             labeling[i, j] = minLabel
#
#     print(labeling)
#
#     # second pass
#     for i in range(len(inputImg)):
#         for j in range(len(inputImg[0])):
#             minLabel = labeling[i, j]
#             if j - 1 >= 0 and inputImg[i][j] == inputImg[i][j - 1]:
#                 if minLabel > labeling[i][j - 1]:
#                     minLabel = labeling[i][j - 1]
#
#             if i - 1 >= 0 and inputImg[i][j] == inputImg[i - 1][ j]:
#                 if minLabel > labeling[i - 1][j]:
#                     minLabel = labeling[i - 1][j]
#
#             if i - 1 >= 0 and j - 1 >= 0 and inputImg[i][j] == inputImg[i - 1][j - 1]:
#                 if minLabel > labeling[i - 1][j - 1]:
#                     minLabel = labeling[i - 1][j - 1]
#
#             if i - 1 >= 0 and j + 1 < len(inputImg[0]) and inputImg[i][j] == inputImg[i - 1][j + 1]:
#                 if minLabel > labeling[i - 1][j + 1]:
#                     minLabel = labeling[i - 1][j + 1]
#
#             if j + 1 < len(inputImg[0]) and inputImg[i][j] == inputImg[i][j + 1]:
#                 if minLabel > labeling[i][j + 1]:
#                     minLabel = labeling[i][j + 1]
#
#             if i + 1 < len(inputImg) and j + 1 < len(inputImg[0]) and inputImg[i][j] == inputImg[i + 1][j + 1]:
#                 if minLabel > labeling[i + 1][j + 1]:
#                     minLabel = labeling[i + 1][j + 1]
#
#             if i + 1 < len(inputImg) and inputImg[i][j] == inputImg[i + 1][j]:
#                 if minLabel > labeling[i + 1][j]:
#                     minLabel = labeling[i + 1][j]
#
#             if i + 1 < len(inputImg) and j - 1 >=0 and inputImg[i][j] == inputImg[i + 1][j - 1]:
#                 if minLabel > labeling[i + 1][j - 1]:
#                     minLabel = labeling[i + 1][j - 1]
#
#             labeling[i][j] = minLabel
#
#     return labeling
#
#
#
#
# if __name__ == "__main__":
#     input1 = [[255, 0, 55, 0],
#              [15, 0, 15, 0],
#              [255, 0, 255, 0],
#              [21, 0, 22, 0],
#              [255, 25, 255, 0],
#              [0, 0, 0, 0],
#              [110, 23, 255, 0],
#              [0, 0, 0, 0],
#              [321, 0, 99, 0]]
#
#     print(connectedComponents(input1))
#
#     # input = [[255, 0, 0, 0],
#     #          [255, 0, 255, 0],
#     #          [255, 0, 255, 0],
#     #          [255, 0, 255, 0],
#     #          [255, 255, 255, 0],
#     #          [0, 0, 0, 0],
#     #          [0, 255, 255, 0],
#     #          [0, 255, 255, 0],
#     #          [255, 0, 0, 0]]
