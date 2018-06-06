from netCDF4 import Dataset
import numpy as np
from concurrent import futures

def create_real(d, t):
    # print(thread)
    original_data = rain_models[d, t, 0, :, :]
    od = np.array(original_data)

    for b1 in range(231):
        for a1 in range(336):
            s1 = np.round(np.sum(od[(5 * b1):(5 * b1) + 5, (5 * a1):(5 * a1) + 5]), 4)
            summing_models[d, t, 0, b1, a1] = s1

def create_models(d, t, m):
    # print(thread)
    try:
        # prediction = "F:/dataset/rain_data/prediction/2d." + str(days[d]) + "-acc/" + str(models[m]) + "/ar" + str(days[d]) + "00.netacc03_" + str(time[t])
        # netcdfFile = Dataset(prediction)
        # rain = netcdfFile.variables['acc03_'][:]
        rain100 = rain_models[d, t, m, :, :]
        r1 = np.array(rain100)

        for b2 in range(231):
            for a2 in range(336):
                s2 = np.round(np.sum(r1[(5 * b2):(5 * b2) + 5, (5 * a2):(5 * a2) + 5]), 4)
                summing_models[d, t, m, b2, a2] = s2
                # print(summing_models[d, t, m, b2, a2])
                # summing_models[d, t, m, :, :] = r1
    except:
        print('exception found')

if __name__ == '__main__':
    # reading netcdf
    netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
    rain_models = netcdf_entire_dataset.variables['rain_models']

    dataset = Dataset("summing_dataset5_5.nc", "w", format="NETCDF4")
    dataset.set_auto_mask(False)

    days = dataset.createDimension('days', 39)
    time = dataset.createDimension('time', 20)
    models = dataset.createDimension('models', 40)
    y_axis = dataset.createDimension('y_axis', 231)
    x_axis = dataset.createDimension('x_axis', 336)

    days = dataset.createVariable('days', "S10", ("days"), zlib=True)
    time = dataset.createVariable('time', "S10", ("time"), zlib=True)
    models = dataset.createVariable('models', "S30", ("models"), zlib=True)

    summing_models = dataset.createVariable("summing_models", "f4", ("days", "time", "models", "y_axis", "x_axis"),
                                            zlib=True)

    print(dataset.variables)


    with futures.ThreadPoolExecutor(max_workers=2) as ex:
        for d in range(39):
            for t in range(20):
                print(d, t, 0, 5)
                ex.submit(create_real, d, t)

    with futures.ThreadPoolExecutor(max_workers=2) as ex2:
        for d in range(39):
            for t in range(20):
                for m in range(39):
                    if m > 0:
                        print(d, t, m, 5)
                        ex2.submit(create_models, d, t, m)


