import csv
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


readData = pd.read_csv('n_complicated_2vlatest.csv', header=None)
imagesArr = []

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
        plt.ylabel('Conceptual Models from 1 to 24')
        plt.xlabel('Conceptual Models from 1 to 24')
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


result = np.zeros(shape=(24,24))

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

tempCheck = rain_models[:20, :10, 0, :, :]
for modelx in range(0, 24):
    for modely in range(0, 24):
        print(modelx, modely)
        idx = 0
        for grid_y in range(1, 45):  # for every y
            for grid_x in range(1, 66):  # for every x
                # print('=================PLACE:', grid_x, grid_y, '=====================')

                if not tempCheck[:,:,grid_y, grid_x].any():
                    pass
                elif imagesArr[modelx][idx] == imagesArr[modely][idx]:
                    result[modelx][modely] += 1
                idx += 1

data_visualization_2dr(result, "")
print(result)



# clustering
def checkSet(p1, p2, pairs):
    newList = [p1, p2]
    newList.sort()
    return tuple(newList) in pairs

def addInTriples(p1, p2, p3, triples):
    tripleList = [p1, p2, p3]
    tripleList.sort()
    triples.add(tuple(tripleList))

threshold = int(input("Initial threshold? "))
while threshold != 0:
    pairs = set()
    # pairs
    for modelx in range(24):
        for modely in range(24):
            if (modelx != modely and result[modelx][modely] >= threshold):
                pairlist = [modelx, modely]
                pairlist.sort()
                pairs.add(tuple(pairlist))

    print(pairs)

    # triples
    triples = set()

    for pair1 in pairs:
        for pair2 in pairs:
            if pair1 != pair2:
                if pair1[0] == pair2[0] and checkSet(pair1[1], pair2[1], pairs):
                    addInTriples(pair1[0], pair1[1], pair2[1], triples)
                elif pair1[1] == pair2[1] and checkSet(pair1[0], pair2[0], pairs):
                    addInTriples(pair1[1], pair1[0], pair2[0], triples)
                elif pair1[0] == pair2[1] and checkSet(pair1[1], pair2[0], pairs):
                    addInTriples(pair1[0], pair1[1], pair2[0], triples)
                elif pair1[1] == pair2[0] and checkSet(pair1[0], pair2[1], pairs):
                    addInTriples(pair1[1], pair1[0], pair2[1], triples)

    print(triples)

    cw = csv.writer(open(str(threshold)+".csv",'w'))

    cw.writerow(list(["Tuples of 2:"]))
    cw.writerow(list(pairs))
    cw.writerow(list(["Tuples of 3:"]))
    cw.writerow(list(triples))

    prev = triples
    count = 4
    while len(prev) > 0 and count < 25:
        newSet = set()
        for triple1 in prev:
            for triple2 in prev:
                for triple3 in prev:
                    if triple1 != triple2 and triple2 != triple3 and triple1 != triple3:
                        newQuad = list(set(triple1).union(triple2).union(triple3))
                        if (len(newQuad) == count):
                            newQuad.sort()
                            newSet.add(tuple(newQuad))

        print(newSet)
        if len(newSet) > 0:
            cw.writerow(list(["Tuples of " + str(count) +":"]))
            cw.writerow(list(newSet))
        prev = newSet
        count += 1

    threshold = int(input("What is the threshold? "))


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import csv
#
# readData = pd.read_csv('n_complicated_2vlatest.csv', header=None)
# imagesArr = []
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
#
# result = np.zeros(shape=(24,24))
#
# for modelx in range(24):
#     for modely in range(24):
#         for idx in range(2860):
#             if imagesArr[modelx][idx] == imagesArr[modely][idx]:
#                 result[modelx][modely] += 1
#
# # data_visualization_2dr(result, "")
# print(result)
#
#
#
# # clustering
# threshold = 1650
# pairs = set()
# # pairs
# for modelx in range(24):
#     for modely in range(24):
#         if (modelx != modely and result[modelx][modely] >= threshold):
#             pairlist = [modelx, modely]
#             pairlist.sort()
#             pairs.add(tuple(pairlist))
#
# print(pairs)
#
# # triples
# triples = set();
# def checkSet(p1, p2):
#     newList = [p1, p2]
#     newList.sort()
#     return tuple(newList) in pairs
#
# def addInTriples(p1, p2, p3):
#     tripleList = [p1, p2, p3]
#     tripleList.sort()
#     triples.add(tuple(tripleList))
#
# for pair1 in pairs:
#     for pair2 in pairs:
#         if pair1 != pair2:
#             if pair1[0] == pair2[0] and checkSet(pair1[1], pair2[1]):
#                 addInTriples(pair1[0], pair1[1], pair2[1])
#             elif pair1[1] == pair2[1] and checkSet(pair1[0], pair2[0]):
#                 addInTriples(pair1[1], pair1[0], pair2[0])
#             elif pair1[0] == pair2[1] and checkSet(pair1[1], pair2[0]):
#                 addInTriples(pair1[0], pair1[1], pair2[0])
#             elif pair1[1] == pair2[0] and checkSet(pair1[0], pair2[1]):
#                 addInTriples(pair1[1], pair1[0], pair2[1])
#
# print(triples)
#
# # quadruples
# quadruples = set()
# for triple1 in triples:
#     for triple2 in triples:
#         for triple3 in triples:
#             if triple1 != triple2 and triple2 != triple3 and triple1 != triple3:
#                 newQuad = list(set(triple1).union(triple2).union(triple3))
#                 if (len(newQuad) == 4):
#                     newQuad.sort()
#                     quadruples.add(tuple(newQuad))
#
# print(quadruples)
#
# # quintuple
# quintuples = set()
# for quadruple1 in quadruples:
#     for quadruple2 in quadruples:
#         for quadruple3 in quadruples:
#             if quadruple1 != quadruple2 and quadruple2 != quadruple3 and quadruple1 != quadruple3:
#                 newQuin = list(set(quadruple1).union(quadruple2).union(quadruple3))
#                 if (len(newQuin) == 5):
#                     newQuin.sort()
#                     quintuples.add(tuple(newQuin))
#
# print(quintuples)
#
# # six
# sixes = set()
# for quintuple1 in quintuples:
#     for quintuple2 in quintuples:
#         for quintuple3 in quintuples:
#             if quintuple1 != quintuple2 and quintuple2 != quintuple3 and quintuple1 != quintuple3:
#                 newSix = list(set(quintuple1).union(quintuple2).union(quintuple3))
#                 if (len(newSix) == 6):
#                     newSix.sort()
#                     sixes.add(tuple(newSix))
#
# print(sixes)
#
# # sevens
# sevens = set()
# for six1 in sixes:
#     for six2 in sixes:
#         for six3 in sixes:
#             if six1 != six2 and six2 != six3 and six1 != six3:
#                 newSeven = list(set(six1).union(six2).union(six3))
#                 if (len(newSeven) == 7):
#                     newSeven.sort()
#                     sevens.add(tuple(newSeven))
#
# print(sevens)
#
# # eights
# eights = set()
# for seven1 in sevens:
#     for seven2 in sevens:
#         for seven3 in sevens:
#             if seven1 != seven2 and seven2 != seven3 and seven1 != seven3:
#                 newEight = list(set(seven1).union(seven2).union(seven3))
#                 if (len(newEight) == 8):
#                     newEight.sort()
#                     eights.add(tuple(newEight))
#
# cw = csv.writer(open(str(threshold)+".csv",'w'))
#
# cw.writerow(list(["pairs:"]))
# cw.writerow(list(pairs))
# cw.writerow(list(["triples:"]))
# cw.writerow(list(triples))
# cw.writerow(list(["quadruples:"]))
# cw.writerow(list(quadruples))
# cw.writerow(list(["quintuples:"]))
# cw.writerow(list(quintuples))
# cw.writerow(list(["sixes:"]))
# cw.writerow(list(sixes))
# cw.writerow(list(["sevens:"]))
# cw.writerow(list(sevens))
# cw.writerow(list(["eights:"]))
# cw.writerow(list(eights))
# print(eights)