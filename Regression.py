import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from netCDF4 import Dataset
import itertools as it
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib as mpl
from matplotlib.pyplot import cm
from sklearn import preprocessing, cross_validation
import math
from sklearn.model_selection import cross_val_score

def preparingData(q, p):
    # reading netcdf
    netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
    rain_models = netcdf_entire_dataset.variables['summing_models']
    days_error_rate_file = netcdf_entire_dataset.variables['days'][:]
    time_error_rate_file = netcdf_entire_dataset.variables['time'][:]
    models_error_rate_file = netcdf_entire_dataset.variables['models'][:]

    size = 24*10
    total_dataset = np.empty(shape=(25,size))

    for i in range(25):
        # print(i)
        x = []
        for j in range(24):
            # print(j)
            for k in range(10):
                print(i, j, k)
                data2 = rain_models[j,k,i,q,p]
                data = np.round(data2, 4)
                # print(data)
                # if data < 5: data = 0
                x.append(data)
                # x.append(list(it.chain.from_iterable(data)))  # flatten the list
        # print(x)
        total_dataset[i] = x #list(it.chain.from_iterable(x))
            # print(total_dataset[i])
        # print(total_dataset)
        # print(total_dataset.transpose().shape)

    col_mean = np.nanmean(total_dataset, axis=0)
    # Find indicies that you need to replace
    inds = np.where(np.isnan(total_dataset))
    # Place column means in the indices. Align the arrays using take
    total_dataset[inds] = np.take(col_mean, inds[1])


    newArray = total_dataset.transpose()
    # print(newArray)
    np.savetxt("Dump/new"+str(q) +"by"+ str(p) +".csv", newArray, delimiter=",",fmt='%10.5f')
    # newArray.tofile("Dump/Coordinate"+str(q) +"by"+ str(p) +".csv", sep=',', format='%10.5f')

    return newArray

# preparingData()


def linearRegression(td):
    train, test = train_test_split(td, test_size=0.2)
    # np.savetxt("Dump/train" + str(q) + "by" + str(p) + ".csv", train, delimiter=",", fmt='%10.5f')
    # np.savetxt("Dump/test" + str(q) + "by" + str(p) + ".csv", test, delimiter=",", fmt='%10.5f')
    clf = LinearRegression(n_jobs=-1)
    # print(clf.coef_)
    clf.fit(train[:, 1:], train[:,0])
    # print(clf.coef_)
    confidence = clf.score(test[:, 1:], test[:,0])
    print(confidence)
    print(clf.coef_)
    # scores = cross_val_score(clf, Train[:, 1:], Train[:,0], cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # plot_result(0, 1)
    #
    predictedValue = clf.predict(test[:, 1:])
    plot_result(predictedValue, test[:,0])

def plot_result(predictedValue, OriginalValue):
    mpl.style.use('seaborn')
    # color = iter(cm.rainbow(np.linspace(0, 1, len(X_train[0]))))

    plt.plot(OriginalValue, color='Green', label='Actual')
    plt.plot(predictedValue, color='Red', label='New Prediction')
    plt.legend()
    # for i in range(len(X_train[0])):
    #     c = next(color)
    #     plt.plot(X_train[:,i], c=c, label='Old Prediction'+str(i))



    plt.show()


for q in range(25, 26):
    for p in range(25, 26):
        td = preparingData(q, p)
        linearRegression(td)

# obj = LinearRegressionClass;
# df = obj.readFromDB(obj, ma=[50, 100, 200])
# obj.load_data(obj, df)