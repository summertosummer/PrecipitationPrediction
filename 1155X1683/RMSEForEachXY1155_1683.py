import numpy as np
import csv
from netCDF4 import Dataset
from sklearn.metrics import *
from numpy import *

# RMSE_result = sqrt(mean_squared_error([5], [49]))
# a = sqrt(power((5 - 49), 2))
# print(RMSE_result, a)

#
# with open('F:/dataset/rain_data/index70.csv') as csvf:
#     ind70 = csv.reader(csvf)
#     indexi70 = list(ind70)
#     index70 = indexi70[0]

#append netcdf
# append_netcdf = Dataset("F:/dataset/new_mae.nc", "a")
# Total_MAE = append_netcdf.variables['Total_MAE']

# reading netcdf
# error_rate_file = "F:/dataset/summing_mae.nc"
# netcdf_error_rate_file = Dataset(error_rate_file)
# models_error_rate_file = netcdf_error_rate_file.variables['models'][:]

# days_error_rate_file = netcdf_error_rate_file.variables['days'][:]
# time_error_rate_file = netcdf_error_rate_file.variables['time'][:]
# models_error_rate_file = netcdf_error_rate_file.variables['models'][:]
# MAE = netcdf_error_rate_file.variables['MAE']

#reading netcdf
netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
rain_model = netcdf_entire_dataset.variables['rain_models']
models_error_rate_file = netcdf_entire_dataset.variables['models'][:]
# days_error_rate_file = netcdf_entire_dataset.variables['days'][:]
# time_error_rate_file = netcdf_entire_dataset.variables['time'][:]
# models_error_rate_file = netcdf_entire_dataset.variables['models'][:]

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
np.seterr(divide='ignore', invalid='ignore')


#creating csv file
check = open('RMSE1155_1683.csv', 'w')
check.truncate()
# writing the headers
check.write(str('Y'))
check.write(', ')
check.write(str('X'))
check.write(', ')
for i in range(1, len(models_error_rate_file)):
    check.write(str(models_error_rate_file[i]))
    check.write(', ')
check.write('\n')

y2 = 0
for z in range(0, 10, 10):
    rain_models = rain_model[:, :, :, z:z + 10, :]
    for y in range(10): # 46 y-coordinates
        # print('model:', i, 'day:', j)
        for x in range(1683): # 67 x-coordinates
            check.write(str(y2))
            check.write(', ')
            check.write(str(x))
            check.write(', ')
            for i in range(1, len(models_error_rate_file)): # for every model
                countArr = np.zeros(shape=(20, 10)) #count array
                sum = 0
                count = 0

                print('Y:', y2, 'X:', x, 'model:', i)
                original_data = np.array(rain_models[:20, :10, 0, y, x]) # real data
                rain100 = np.array(rain_models[:20, :10, i, y, x]) # model data
                # rain100[rain100>30000] = np.nan

                # print(sqrt(power(abs(original_data - rain100), 2)))
                # print(original_data)
                # print(rain100)
                a = np.array(power((original_data - rain100), 2)) # square of the difference
                # print(a[0, 0])

                # print(len(a), len(a[0]))
                # print(a[2,3])
                # print(np.nanmin(a), np.nanmax(a))
                # a[a > 30000] = np.nan
                if not (np.isnan(a).all() and np.isinf(a).all):

                    # print('yes')
                    # print(MAE[j, k, i, :, :])
                    # sum = sum + np.array(a)
                    # print(sum)
                    countArr = countArr + 1 #counting for all values

                    mask = ((original_data == 0) & (rain100 == 0)) | (a == np.inf) | (a == np.nan) # if both real and prediction model has value zero
                    # print('mask found:', mask)
                    countArr[mask] = countArr[mask] - 1 # doing (-1) for not counting zero values
                    # max = np.round(np.nanmax(a), 4)
                    # min = np.round(np.nanmin(a), 4)
                    # print(min,max)

                    a[a == np.nan] = 0
                    a[a == np.inf] = 0

                    # print(a)
                    sum = np.nansum(a) #summing all non-nan values
                        # print(sum)
                    count= np.sum(countArr)
                    if sum == 0 and count == 0:
                        avg = 0
                    else:
                        avg = sum / count # root mean square error
                    print(sum, count, avg)

                    check.write(str(avg))
                    check.write(', ')
            check.write('\n')
        y2 += 1
check.close()

    # # print(sum)
    # avg = np.divide(sum, count)
    # # print(avg)
    # Total_MAE[i, :, :] = avg
    # print(Total_MAE[i, :, :])

# append_netcdf.close()
