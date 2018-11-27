# import math
# def deliveryPlan (numDestinations, allLocations, numDeliveries):
#     dict = {}
#     for i in range(0, numDestinations):
#         temp = allLocations[i]
#         distance = math.sqrt((temp[0]**2) + (temp[1]**2))
#         dict[i] = distance
#
#     sorted_key = sorted(dict, key = dict.get)
#
#     return [allLocations[i] for i in sorted_key[:numDeliveries]]
#
# print(deliveryPlan(3, [[1, 2], [3, 4], [1, -1]], 2))
# print(deliveryPlan(6, [[3, 6], [2, 4], [5, 3], [2, 7], [1, 8], [7, 9]], 3))

#
# # from copy import deepcopy
import numpy as np
def minimumDistanceTraverse(numRows, numColumns, lot):
    dp = np.empty(shape=(numRows,numColumns))
    dp[0][0] = 0
    for j in range(numRows):
        for i in range(numColumns):
            if lot[j][i] == 9:
                return min(dp[j][i-1], dp[j-1][i])

            if (i > 0 and lot[j][i] == 1):
                # print(dp[j, i-1])
                dp[j, i] = dp[j][i-1] + 1
            elif (j > 0 and lot[j][i] == 1):
                # print(dp[j - 1, i])
                dp[j][i] = dp[j - 1][i] + 1
            else:
                dp[j][i] = 0
    # print(dp)
    return 0

print(minimumDistanceTraverse(3, 3, [[1,0,0], [1,0,0], [1,9,1]]))



























# def minimumDemolitionDistance(numRows, numColumns, lot):
#     nodes = [[0, 0, -1]]  # x,y, distance
#     checked = set([])
#     m = numRows
#     n = numColumns
#
#     while nodes:
#         node = nodes.pop(0)
#         x, y, d = node
#         uid = n * x + y
#
#         if uid in checked:
#             continue
#
#         if lot[x][y] == 9:
#             return d + 1
#
#         newd = d + 1
#
#         # add right
#         if y + 1 < n:
#             uid = n * x + (y + 1)
#             if (uid not in checked) and (lot[x][y + 1] != 0):
#                 nodes.append([x, y + 1, newd])
#         # add left
#         if y - 1 >= 0:
#             uid = n * x + (y - 1)
#             if uid not in checked and (lot[x][y - 1] != 0):
#                 nodes.append([x, y - 1, newd])
#         # add bottom
#         if x + 1 < m:
#             uid = n * (x + 1) + y
#             if uid not in checked and (lot[x + 1][y] != 0):
#                 nodes.append([x + 1, y, newd])
#         # add top
#         if x - 1 >= 0:
#             uid = n * (x - 1) + y
#             if uid not in checked and (lot[x - 1][y] != 0):
#                 nodes.append([x - 1, y, newd])
#     return -1
#
#
# lot = [
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [0, 0, 0, 1],
#     [1, 1, 9, 1]
# ]
# print (minimumDemolitionDistance(len(lot), len(lot[0]), lot))