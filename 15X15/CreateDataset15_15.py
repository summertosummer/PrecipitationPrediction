from netCDF4 import Dataset
import glob
from os.path import basename
from pyhdf.SD import *
import numpy as np

#reading netcdf
netcdf_entire_dataset = Dataset("entire_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['rain_models']

dataset = Dataset("summing_dataset15_15.nc", "w", format="NETCDF4")
dataset.set_auto_mask(False)

days = dataset.createDimension('days', 39)
time = dataset.createDimension('time', 20)
models = dataset.createDimension('models', 40)
y_axis = dataset.createDimension('y_axis', 77)
x_axis = dataset.createDimension('x_axis', 112)

days = dataset.createVariable('days', "S10",("days"),zlib=True)
time = dataset.createVariable('time', "S10",("time"),zlib=True)
models = dataset.createVariable('models', "S30",("models"),zlib=True)

summing_models = dataset.createVariable("summing_models","f4",("days","time","models","y_axis","x_axis"),zlib=True)

print(dataset.variables)
#
# for dimname in dataset.dimensions.keys():
#     dim = dataset.dimensions[dimname]
#     print (dimname, len(dim), dim.isunlimited())
#
# # writing days
# i = 0
# for day in glob.glob("F:/dataset/rain_data/verification/*"):
#     # print(basename(day))
#     days[i] = basename(day)
#     i = i + 1
# print(days[:])
#
# #writing times
# i = 0
# for second in glob.glob("F:/dataset/rain_data/verification/20160421/*"):
#     # print(basename(second).split('_')[1]))
#     time[i] = basename(second).split('_')[1]
#     i = i + 1
# print(time[:])
#
# # writing models
# models[0] = 'verification'
# i = 1
# arr = []
# arr.insert(0, 'time')
# for pred_date in glob.glob("F:/dataset/rain_data/prediction/*"):
#     # print(pred_date)
#     for core in glob.glob(str(pred_date) + "/*"):
#         # print(core)
#         string = str(basename(core))
#         if not string in arr:
#             # print(basename(core))
#             arr.append(str(basename(core)))
#             models[i] = str(basename(core))
#             i = i + 1
# print(models[:])

for d in range(39):
    for t in range(20):
        print(d, t, 0)
        # reading hdf file
        # verification = "F:/dataset/rain_data/verification/" + str(days[d]) + "/ar" + str(days[d]) + "00.hdfacc03_" + str(time[t])
        # hdfFile = SD(verification, SDC.READ)
        # datasets_dic = hdfFile.datasets()
        #
        # sds_obj = hdfFile.select('acc03_')  # select sds
        #
        # data = sds_obj.get()  # get sds data
        original_data = rain_models[d, t, 0, :, :]

        # rescaling the data
        # for key, value in sds_obj.attributes().items():
        #     if key == 'max':
        #         maxValue = value
        #     if key == 'min':
        #         minValue = value
        #
        # scalef = (maxValue - minValue) / 65534.0
        # original_data = scalef * (original_data + 32767) + minValue
        od = np.array(original_data)

        for b1 in range(77):
            for a1 in range(112):
                s1 = np.round(np.sum(od[(15 * b1):(15 * b1) + 15, (15 * a1):(15 * a1) + 15]), 4)
                summing_models[d, t, 0, b1, a1] = s1
                # print(summing_models[d, t, 0, b1, a1])
        # print(original_data[:][:])
        # print(rain_models[d][t][0][:][:])

for d in range(39):
    for t in range(20):
        for m in range(39):
            if m > 0:
                print(d, t, m)
                try:
                    # prediction = "F:/dataset/rain_data/prediction/2d." + str(days[d]) + "-acc/" + str(models[m]) + "/ar" + str(days[d]) + "00.netacc03_" + str(time[t])
                    # netcdfFile = Dataset(prediction)
                    # rain = netcdfFile.variables['acc03_'][:]
                    rain100 = rain_models[d, t, m, :, :]
                    r1 = np.array(rain100)

                    for b2 in range(77):
                        for a2 in range(112):
                            s2 = np.round(np.sum(r1[(15 * b2):(15 * b2) + 15, (15 * a2):(15 * a2) + 15]), 4)
                            summing_models[d, t, m, b2, a2] = s2
                            # print(summing_models[d, t, m, b2, a2])
                    # summing_models[d, t, m, :, :] = r1
                except:
                    continue


