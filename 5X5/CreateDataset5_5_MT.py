# from netCDF4 import Dataset
# import glob
# from os.path import basename
# # from pyhdf.SD import *
# import numpy as np
# # import _thread
# import threading
#
# #reading netcdf
# netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
# rain_models = netcdf_entire_dataset.variables['rain_models']
#
# dataset = Dataset("summing_dataset5_5_v2.nc", "w", format="NETCDF4")
# dataset.set_auto_mask(False)
#
# days = dataset.createDimension('days', 39)
# time = dataset.createDimension('time', 20)
# models = dataset.createDimension('models', 40)
# y_axis = dataset.createDimension('y_axis', 231)
# x_axis = dataset.createDimension('x_axis', 336)
#
# days = dataset.createVariable('days', "S10",("days"),zlib=True)
# time = dataset.createVariable('time', "S10",("time"),zlib=True)
# models = dataset.createVariable('models', "S30",("models"),zlib=True)
#
# summing_models = dataset.createVariable("summing_models","f4",("days","time","models","y_axis","x_axis"),zlib=True)
#
# print(dataset.variables)
# #
# # for dimname in dataset.dimensions.keys():
# #     dim = dataset.dimensions[dimname]
# #     print (dimname, len(dim), dim.isunlimited())
# #
# # # writing days
# # i = 0
# # for day in glob.glob("F:/dataset/rain_data/verification/*"):
# #     # print(basename(day))
# #     days[i] = basename(day)
# #     i = i + 1
# # print(days[:])
# #
# # #writing times
# # i = 0
# # for second in glob.glob("F:/dataset/rain_data/verification/20160421/*"):
# #     # print(basename(second).split('_')[1]))
# #     time[i] = basename(second).split('_')[1]
# #     i = i + 1
# # print(time[:])
# #
# # # writing models
# # models[0] = 'verification'
# # i = 1
# # arr = []
# # arr.insert(0, 'time')
# # for pred_date in glob.glob("F:/dataset/rain_data/prediction/*"):
# #     # print(pred_date)
# #     for core in glob.glob(str(pred_date) + "/*"):
# #         # print(core)
# #         string = str(basename(core))
# #         if not string in arr:
# #             # print(basename(core))
# #             arr.append(str(basename(core)))
# #             models[i] = str(basename(core))
# #             i = i + 1
# # print(models[:])
#
# def create_real(thread, d, t):
#     print(thread)
#     original_data = rain_models[d, t, 0, :, :]
#     od = np.array(original_data)
#
#     for b1 in range(231):
#         for a1 in range(336):
#             s1 = np.round(np.sum(od[(5 * b1):(5 * b1) + 5, (5 * a1):(5 * a1) + 5]), 4)
#             summing_models[d, t, 0, b1, a1] = s1
#
# def create_models(thread, d, t, m):
#     print(thread)
#     try:
#         # prediction = "F:/dataset/rain_data/prediction/2d." + str(days[d]) + "-acc/" + str(models[m]) + "/ar" + str(days[d]) + "00.netacc03_" + str(time[t])
#         # netcdfFile = Dataset(prediction)
#         # rain = netcdfFile.variables['acc03_'][:]
#         rain100 = rain_models[d, t, m, :, :]
#         r1 = np.array(rain100)
#
#         for b2 in range(231):
#             for a2 in range(336):
#                 s2 = np.round(np.sum(r1[(5 * b2):(5 * b2) + 5, (5 * a2):(5 * a2) + 5]), 4)
#                 summing_models[d, t, m, b2, a2] = s2
#                 # print(summing_models[d, t, m, b2, a2])
#                 # summing_models[d, t, m, :, :] = r1
#     except:
#         print('exception found')
#
# thread_list1 = []
# for d in range(39):
#     for t in range(20):
#         print(d, t, 0, 5)
#         # Create two threads as follows
#         t = threading.Thread(target=create_real, args=('Thread: '+ str(d) + str(t), d, t,))
#         thread_list1.append(t)
#         # t.daemon = True
#         # t.start()
#         # try:
#         #     _thread.start_new_thread(create_real, ('Thread: '+ str(d) + str(t), d, t))
#         # except:
#         #     print("Error: unable to start thread")
#
#
#         # reading hdf file
#         # verification = "F:/dataset/rain_data/verification/" + str(days[d]) + "/ar" + str(days[d]) + "00.hdfacc03_" + str(time[t])
#         # hdfFile = SD(verification, SDC.READ)
#         # datasets_dic = hdfFile.datasets()
#         #
#         # sds_obj = hdfFile.select('acc03_')  # select sds
#         #
#         # data = sds_obj.get()  # get sds data
#         # ###################
#         # original_data = rain_models[d, t, 0, :, :]
#         # ####################
#         # rescaling the data
#         # for key, value in sds_obj.attributes().items():
#         #     if key == 'max':
#         #         maxValue = value
#         #     if key == 'min':
#         #         minValue = value
#         #
#         # scalef = (maxValue - minValue) / 65534.0
#         # original_data = scalef * (original_data + 32767) + minValue
#         #     ##################
#         # od = np.array(original_data)
#         #     ####################
#
#         # for b1 in range(231):
#         #     for a1 in range(336):
#         #         s1 = np.round(np.sum(od[(5 * b1):(5 * b1) + 5, (5 * a1):(5 * a1) + 5]), 4)
#         #         summing_models[d, t, 0, b1, a1] = s1
#         #         # print(summing_models[d, t, 0, b1, a1])
#         # # print(original_data[:][:])
#         # # print(rain_models[d][t][0][:][:])
#
# thread_list2 = []
# for d in range(39):
#     for t in range(20):
#         for m in range(39):
#             if m > 0:
#                 print(d, t, m, 5)
#                 t = threading.Thread(target=create_models, args=('Thread: ' + str(d) + str(t), d, t, m,))
#                 thread_list2.append(t)
#                 # t.daemon = True
#                 # t.start()
#
#                 # try:
#                 #     _thread.start_new_thread(create_models, ('Thread: ' + str(d) + str(t), d, t, m))
#                 # except:
#                 #     print("Error: unable to start thread")
#
# for thread in thread_list1:
#     thread.start()
#     thread.join()
# for thread in thread_list2:
#     thread.start()
#     thread.join()
#

import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    print('start')
    if n % 2 == 0:
        print('end')
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            print('end')
            return False
    print('end')
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for prime in executor.map(is_prime, PRIMES):
            print(prime)

if __name__ == '__main__':
    main()