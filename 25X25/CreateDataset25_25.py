from netCDF4 import Dataset
import glob
from os.path import basename
from pyhdf.SD import *
import numpy as np

dataset = Dataset("F:/dataset/summing_dataset.nc", "w", format="NETCDF4")
dataset.set_auto_mask(False)

days = dataset.createDimension('days', 39)
time = dataset.createDimension('time', 20)
models = dataset.createDimension('models', 40)
y_axis = dataset.createDimension('y_axis', 46)
x_axis = dataset.createDimension('x_axis', 67)

days = dataset.createVariable('days', "S10",("days"),zlib=True)
time = dataset.createVariable('time', "S10",("time"),zlib=True)
models = dataset.createVariable('models', "S30",("models"),zlib=True)

summing_models = dataset.createVariable("summing_models","f4",("days","time","models","y_axis","x_axis"),zlib=True)

print(dataset.variables)
#
# for dimname in dataset.dimensions.keys():
#     dim = dataset.dimensions[dimname]
#     print (dimname, len(dim), dim.isunlimited())

# writing days
i = 0
for day in glob.glob("F:/dataset/rain_data/verification/*"):
    # print(basename(day))
    days[i] = basename(day)
    i = i + 1
print(days[:])

#writing times
i = 0
for second in glob.glob("F:/dataset/rain_data/verification/20160421/*"):
    # print(basename(second).split('_')[1]))
    time[i] = basename(second).split('_')[1]
    i = i + 1
print(time[:])

# writing models
models[0] = 'verification'
i = 1
arr = []
arr.insert(0, 'time')
for pred_date in glob.glob("F:/dataset/rain_data/prediction/*"):
    # print(pred_date)
    for core in glob.glob(str(pred_date) + "/*"):
        # print(core)
        string = str(basename(core))
        if not string in arr:
            # print(basename(core))
            arr.append(str(basename(core)))
            models[i] = str(basename(core))
            i = i + 1
print(models[:])

for d in range(len(days)):
    for t in range(len(time)):
        print(days[d], time[t], models[0])
        # reading hdf file
        verification = "F:/dataset/rain_data/verification/" + str(days[d]) + "/ar" + str(days[d]) + "00.hdfacc03_" + str(time[t])
        hdfFile = SD(verification, SDC.READ)
        datasets_dic = hdfFile.datasets()

        sds_obj = hdfFile.select('acc03_')  # select sds

        data = sds_obj.get()  # get sds data
        original_data = data[0, :, :]

        # rescaling the data
        for key, value in sds_obj.attributes().items():
            if key == 'max':
                maxValue = value
            if key == 'min':
                minValue = value

        scalef = (maxValue - minValue) / 65534.0
        original_data = scalef * (original_data + 32767) + minValue
        od = np.array(original_data)

        for b1 in range(46):
            for a1 in range(67):
                s1 = np.round(np.sum(od[(25 * b1):(25 * b1) + 25, (25 * a1):(25 * a1) + 25]), 4)
                summing_models[d, t, 0, b1, a1] = s1
                # print(summing_models[d, t, 0, b1, a1])
        # print(original_data[:][:])
        # print(rain_models[d][t][0][:][:])

for d in range(len(days)):
    for t in range(len(time)):
        for m in range(len(models)):
            if m > 0:
                print(days[d], time[t], models[m])
                try:
                    prediction = "F:/dataset/rain_data/prediction/2d." + str(days[d]) + "-acc/" + str(models[m]) + "/ar" + str(days[d]) + "00.netacc03_" + str(time[t])
                    netcdfFile = Dataset(prediction)
                    rain = netcdfFile.variables['acc03_'][:]
                    rain100 = rain[0, :, :]
                    r1 = np.array(rain100)

                    for b2 in range(46):
                        for a2 in range(67):
                            s2 = np.round(np.sum(r1[(25 * b2):(25 * b2) + 25, (25 * a2):(25 * a2) + 25]), 4)
                            summing_models[d, t, m, b2, a2] = s2
                            # print(summing_models[d, t, m, b2, a2])
                    # summing_models[d, t, m, :, :] = r1
                except:
                    continue


