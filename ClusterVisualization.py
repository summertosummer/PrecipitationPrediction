from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

readData = pd.read_csv('n_complicated_2vlatest.csv', header=None)
imagesArr = []
newclusters = ["(0, 1, 2, 3, 4, 7, 9, 10)","(7, 9, 10, 14, 18, 19, 21, 23)","(7, 9, 10, 14, 16, 18, 19, 23)","(7, 9, 10, 14, 16, 17, 19, 23)"
                ]
clusters = []
for n in newclusters:
    clusters.append(list(eval(n)))
print(clusters)

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
        plt.ylabel('Vertical Grid')
        plt.xlabel('Horizontal Grid')

        for i, txt in enumerate(c):
            if txt <= 4:
                plt.annotate(int(txt), (y[i], x[i]))
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()

for i in range(24):
    temp = pd.to_numeric(np.array(readData[i])[:])
    temp[temp <=0] = 0
    temp[temp > 0] = 1
    imagesArr.append(temp)

print(np.array(imagesArr))

result = np.array(["0000" for _ in range(2860)])
countCluster = 0
dictVal = 5

dict = {}

for x in range(1, 5):
    if x == 1: dict["1000"] = x
    if x == 2: dict["0100"] = x
    if x == 3: dict["0010"] = x
    if x == 4: dict["0001"] = x
print(dict)

for cidx in range(0, len(clusters)):
    cluster = clusters[cidx]
    countCluster += 1
    print(countCluster)
    for idx in range(2860):
        cValue = imagesArr[cluster[0]][idx]
        flag = True
        for c in cluster:
            if imagesArr[c][idx] != cValue:
                flag = False
                break

        if flag:
            # 1D result array is basically the 2D geographical map
            if result[idx] == "0000": # result index is zero when there was no cluster before, i.e., no overlap
                if countCluster == 1: result[idx] = "1000"
                if countCluster == 2: result[idx] = "0100"
                if countCluster == 3: result[idx] = "0010"
                if countCluster == 4: result[idx] = "0001"
            else: # otherwise overlaps. adding all overlap clusters num with a small float value to make it unique
                temp = list(result[idx])
                if countCluster == 1: temp[0] = '1'
                if countCluster == 2: temp[1] = '1'
                if countCluster == 3: temp[2] = '1'
                if countCluster == 4: temp[3] = '1'
                result[idx] = "".join(temp)

            if result[idx] not in dict:
                dict[result[idx]] = dictVal
                dictVal += 1

print(dict)

finalResult = np.zeros(shape=(2860,1))
for idx in range(2860):
    tmp = result[idx]
    if tmp != "0000":
        finalResult[idx] = dict[tmp]



#removing areas where no rainfall in the real data

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

tempCheck = rain_models[:20, :10, 0, :, :]
f_index = 0
for grid_y in range(1, 45):  # for every y
    for grid_x in range(1, 66):  # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')
        if not tempCheck[:,:,grid_y,grid_x].any():
            finalResult[f_index] = 0
        f_index += 1

print(finalResult.reshape((44, 65)))
data_visualization_2dr(finalResult.reshape((44, 65)), "Clusters Combined")


























# from netCDF4 import Dataset
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# class Node:
#     # c1 = False
#     # c2 = False
#     # c3 = False
#     # c4 = False
#
#     def __init__(self, c1, c2, c3, c4):
#         self.c1 = c1
#         self.c2 = c2
#         self.c3 = c3
#         self.c4 = c4
#
#     # def setc1(self, b):
#     #     self.c1 = b
#     #
#     # def setc2(self, b):
#     #     self.c2 = b
#     #
#     # def setc3(self, b):
#     #     self.c3 = b
#     #
#     # def setc4(self, b):
#     #     self.c4 = b
#
#     def __hash__(self):
#         return hash((self.c1, self.c2, self.c3, self.c4))
#
#     def __eq__(self, other):
#         return (self.c1, self.c2, self.c3, self.c4) == (other.c1, other.c2, other.c3, other.c4)
#
#     # def __ne__(self, other):
#     #     # Not strictly necessary, but to avoid having both x==y and x!=y
#     #     # True at the same time
#     #     return not (self == other)
#
# readData = pd.read_csv('n_complicated_2vlatest.csv', header=None)
# imagesArr = []
#
# newclusters = ["(0, 1, 2, 3, 4, 7, 9, 10)","(7, 9, 10, 14, 18, 19, 21, 23)","(7, 9, 10, 14, 16, 18, 19, 23)","(7, 9, 10, 14, 16, 17, 19, 23)"
#                 ]
# clusters = []
#
# for n in newclusters:
#     clusters.append(list(eval(n)))
#
# print(clusters)
#
# #2D Visualizaiton
# def data_visualization_2dr(w_data, title, i=0, visualize=True):
#     if visualize:
#         plt.axis([0, len(w_data[0]), 0, len(w_data)])
#         # w_data[w_data >= 0] = 0
#         # w_data[w_data >= 100] = 0
#         x, y = w_data.nonzero()
#         # x = range(0, 65)
#         # y = range(0, 44)
#         c = w_data[x, y]
#         plt.scatter(y[:], x[:], c=c[:], cmap='jet')
#         plt.title(title)
#         plt.colorbar()
#         plt.ylabel('Vertical Grid')
#         plt.xlabel('Horizontal Grid')
#
#         for i, txt in enumerate(c):
#             if txt <= 4:
#                 plt.annotate(int(txt), (y[i], x[i]))
#         # plt.savefig('com/fig' + str(i) + '.png')
#         # plt.clim(-5, 0)
#         plt.show()
#         plt.close()
#
# for i in range(24):
#     temp = pd.to_numeric(np.array(readData[i])[:])
#     temp[temp <=0] = 0
#     temp[temp > 0] = 1
#     imagesArr.append(temp)
#
# print(np.array(imagesArr))
#
# result = np.array([Node(False, False, False, False) for _ in range(2860)])
# countCluster = 0
# dictVal = 5
#
# dict = {}
#
# for x in range(1, 5):
#     tempObj = Node(False, False, False, False)
#     if x == 1: tempObj.c1 = True
#     if x == 2: tempObj.c2 = True
#     if x == 3: tempObj.c3 = True
#     if x == 4: tempObj.c4 = True
#
#     dict[tempObj] = x
#
# print(dict)
#
# for cidx in range(0, len(clusters)):
#     cluster = clusters[cidx]
#     countCluster += 1
#     print(countCluster)
#     for idx in range(2860):
#         cValue = imagesArr[cluster[0]][idx]
#         flag = True
#         for c in cluster:
#             if imagesArr[c][idx] != cValue:
#                 flag = False
#                 break
#
#
#         if flag:
#             # 1D result array is basically the 2D geographical map
#             if result[idx] == Node(False, False, False, False): # result index is zero when there was no cluster before, i.e., no overlap
#                 newObj = Node(False, False, False, False)
#                 if countCluster == 1: newObj.c1 = True
#                 if countCluster == 2: newObj.c2 = True
#                 if countCluster == 3: newObj.c3 = True
#                 if countCluster == 4: newObj.c4 = True
#                 result[idx] = newObj # giving a new cluster number
#             else: # otherwise overlaps. adding all overlap clusters num with a small float value to make it unique
#                 temp = result[idx]
#                 if countCluster == 1: temp.c1 = True
#                 if countCluster == 2: temp.c2 = True
#                 if countCluster == 3: temp.c3 = True
#                 if countCluster == 4: temp.c4 = True
#                 result[idx] = temp
#
#             if result[idx] not in dict.keys():
#                 dict[result[idx]] = dictVal
#                 dictVal += 1
#
# if Node(False, True, True, False) in dict.keys():
#     print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
#
#         # if flag:
#         #     # 1D result array is basically the 2D geographical map
#         #     if result[idx] == 0: # result index is zero when there was no cluster before, i.e., no overlap
#         #         result[idx] = countCluster # giving a new cluster number
#         #     else: # otherwise overlaps. adding all overlap clusters num with a small float value to make it unique
#         #         result[idx] = (result[idx] * countCluster) + (result[idx] + countCluster)
#
# if Node(False, True, True, False) in dict.keys():
#     print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
#
# finalResult = np.zeros(shape=(2860,1))
# for idx in range(2860):
#     tmp = result[idx]
#     if tmp != Node(False, False, False, False):
#         # print(tmp.c1,tmp.c2,tmp.c3,tmp.c4)
#         if tmp in dict.keys():
#             # print(tmp.c1, tmp.c2, tmp.c3, tmp.c4)
#             # print("$$$$$$$$$$$$$$$%^&&&&&&&&&&&&&&&&&&&&&&&&&&&&^^^^^^")
#             finalResult[idx] = dict[tmp]
#
#
#
# #removing areas where no rainfall in the real data
#
# #access netcdf data file
# netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
# rain_models = netcdf_entire_dataset.variables['summing_models']
#
# tempCheck = rain_models[:20, :10, 0, :, :]
# f_index = 0
# for grid_y in range(1, 45):  # for every y
#     for grid_x in range(1, 66):  # for every x
#         print('=================PLACE:', grid_x, grid_y, '=====================')
#         if not tempCheck[:,:,grid_y,grid_x].any():
#             finalResult[f_index] = 0
#         f_index += 1
#
# print(finalResult.reshape((44, 65)))
# data_visualization_2dr(finalResult.reshape((44, 65)), "Clusters Combined")