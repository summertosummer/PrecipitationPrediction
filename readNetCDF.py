from netCDF4 import Dataset
import numpy as np

# reading netcdf
netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']
days_error_rate_file = netcdf_entire_dataset.variables['days'][:]
time_error_rate_file = netcdf_entire_dataset.variables['time'][:]
models_error_rate_file = netcdf_entire_dataset.variables['models'][:]

#     rain_models(day, time, models, y, x)
print(rain_models[1, 1, 1, 1, 1])
print(rain_models[1, 1, 1, :, :])