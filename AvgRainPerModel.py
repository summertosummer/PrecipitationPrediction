from netCDF4 import Dataset
import numpy as np

#reading netcdf
netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

# imagesArr = []
# for i in range(1, 25):
minmaxArray = []
for y in range(1, 45):  # 46 y-coordinates
    for x in range(1, 66):  # 67 x-coordinates
        temp = []
        print(y, x)
        for i in range(1, 25):  # for every model
            data = np.array(rain_models[:20, :10, i, y, x])
            data[data == np.nan] = 0
            data[data == np.inf] = 0
            data[data > 100000] = 0
            temp.append(np.nanmean(data))
        minmaxArray.append(temp)

np.savetxt('avgRainPerModel_reshaped.csv', minmaxArray, delimiter=',', fmt='%10.5f')