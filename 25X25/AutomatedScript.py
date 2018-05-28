import pickle
import tensorflow as tf
from netCDF4 import Dataset
import pickle
import numpy as np
import csv
import Orange as og
import math
import itertools as it
from Orange.data import Domain, Table
import random
from Orange.projection import PCA
#import matplotlib.pyplot as plt
#import matplotlib as mpl
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

netcdf_entire_dataset = Dataset("summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

dfMAE = pd.read_csv('MAE.csv', header=None)
dfRMSE = pd.read_csv('RMSE.csv', header=None)
'''
def plot_result(predictedValue, OriginalValue):
    mpl.style.use('seaborn')
    plt.plot(OriginalValue.X, 'o')
    plt.plot(OriginalValue.Y, color='Green', label='Actual')
    plt.plot(predictedValue, color='Red', label='New Prediction')
    plt.legend()
    plt.show()

def plot_input(x, y):
    mpl.style.use('seaborn')
    plt.plot(x, 'o')
    plt.plot(y, color='red', label='Real')
    plt.legend()
    plt.show()
'''
def create_training_and_testing_data(grid_x, grid_y):
    data_x = []
    data_y = []
    tr_count = 0
    for i in range(20):
        for j in range(10):
            x = []
            for k in range(1, 24):
                # print('model: ', k)
                b = rain_models[i, j, k, grid_y-1:grid_y+2, grid_x-1:grid_x+2]
                rain100 = np.array(b)
                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, grid_y, grid_x]
            rainR = bt

            data_y.append(rainR)
            data_x.append(list(it.chain.from_iterable(x)))

    return data_x, data_y

def run_models(grid_y, grid_x):
    X, Y = create_training_and_testing_data(grid_x, grid_y)
    data = Table(X, Y)
    # print(data.Y)
    # np.savetxt('data/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(data), delimiter=',', fmt='%10.5f')
    # print(out_data.domain)
    # print(out_data.Y)

    feature_method = og.preprocess.score.UnivariateLinearRegression()
    selector = og.preprocess.SelectBestFeatures(method=feature_method, k=50)
    out_data2 = selector(data)
    # plot_input(out_data2.X, out_data2.Y)
    # print(out_data2.domain)

    pca = PCA(n_components=5)
    model = pca(out_data2)
    out_data = model(out_data2)
    # print(out_data.domain)

    test = og.data.Table(out_data.domain, random.sample(out_data, 60))
    train = og.data.Table(out_data.domain, [d for d in out_data if d not in test])

    lin = og.regression.linear.LinearRegressionLearner()
    rf = og.regression.random_forest.RandomForestRegressionLearner()
    nnr = og.regression.NNRegressionLearner()
    svm = og.regression.SVRLearner()
    knn = KNeighborsRegressor(n_neighbors=7)

    learners = [lin, rf, nnr, svm]
    regressors = [learner(train) for learner in learners]
    knn.fit(train.X, train.Y)

    with open("models/"+str(grid_x)+str(grid_y)+"_lin.pickle", "wb") as f:
        pickle.dump(lin, f)
    with open("models/"+str(grid_x)+str(grid_y)+"_rf.pickle", "wb") as f:
        pickle.dump(rf, f)
    with open("models/"+str(grid_x)+str(grid_y)+"_nnr.pickle", "wb") as f:
        pickle.dump(nnr, f)
    with open("models/"+str(grid_x)+str(grid_y)+"_svm.pickle", "wb") as f:
        pickle.dump(svm, f)
    with open("models/"+str(grid_x)+str(grid_y)+"_knn.pickle", "wb") as f:
        pickle.dump(knn, f)

    # print((r(test)[0] for r in regressors))
    linPredict = regressors[0](test)
    rfPredict = regressors[1](test)
    nnrPredict = regressors[2](test)
    svmPredict = regressors[3](test)
    knnPredict = knn.predict(test.X)

    predictions = []
    predictions.append(linPredict)
    predictions.append(rfPredict)
    predictions.append(nnrPredict)
    predictions.append(svmPredict)
    predictions.append(knnPredict)

    # print(knnPredict)

    # print("y   ", " ".join("%5s" % l.name for l in regressors))
    # for d in test:
    #     print(("{:<5}" + " {:5.1f}" * len(regressors)).format(d.get_class(), *(r(d)[0] for r in regressors)))

    # res = og.evaluation.CrossValidation(test, learners, k=10)
    # rmse = og.evaluation.RMSE(res)
    # mae = og.evaluation.MAE(res)
    # r2 = og.evaluation.R2(res)

    rmse = []
    mae = []
    rmse.append(math.sqrt(mean_squared_error(test.Y, linPredict)))
    rmse.append(math.sqrt(mean_squared_error(test.Y, rfPredict)))
    rmse.append(math.sqrt(mean_squared_error(test.Y, nnrPredict)))
    rmse.append(math.sqrt(mean_squared_error(test.Y, svmPredict)))
    rmse.append(math.sqrt(mean_squared_error(test.Y, knnPredict)))

    mae.append(mean_absolute_error(test.Y, linPredict))
    mae.append(mean_absolute_error(test.Y, rfPredict))
    mae.append(mean_absolute_error(test.Y, nnrPredict))
    mae.append(mean_absolute_error(test.Y, svmPredict))
    mae.append(mean_absolute_error(test.Y, knnPredict))

    return np.array(mae), np.array(rmse), np.array(predictions), test

def best_mae(minMAE, grid_y, grid_x):
    # print('min MAE:', minMAE)
    df1 = dfMAE[(dfMAE[0] == str(grid_y)) & (dfMAE[1] == str(' ') + str(grid_x))]

    flag = True
    for z in range(2, 26):
        value = pd.to_numeric(dfMAE[z][df1.index], errors='coerce')
        # print(value.values[0])
        if value.values[0] < minMAE:
            flag = False
            break
    return flag

def best_rmse(minRMSE, grid_y, grid_x):
    # print('min rmse:', minRMSE)
    df1 = dfRMSE[(dfRMSE[0] == str(grid_y)) & (dfRMSE[1] == str(' ')+str(grid_x))]

    flag = True
    for z in range(2, 26):
        value = pd.to_numeric(dfRMSE[z][df1.index], errors='coerce')
        # print(value.values[0])
        if value.values[0] < minRMSE:
            flag = False
            break
    return flag

total = 0
countMAE = 0
countRMSE = 0
for grid_y in range (1, 45):
    for grid_x in range (1, 66):
        print('=================PLACE:', grid_x, grid_y, '=====================')

        flag = True
        for _ in range(5):
            try:
                mae, rmse, predictions, test = run_models(grid_y, grid_x)
                minRMSE = np.amin(rmse)
                total += 1
                if best_rmse(minRMSE, grid_y, grid_x):
                    print('found the best')
                    break
            except:
                flag = False
                pass

        # print("Learner  RMSE  MAE  R2")
        # for i in range(len(learners)):
        #     print("{:8s} {:.2f} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], mae[i], r2[i]))

        if flag:
            minMAE = np.amin(mae)
            minRMSE = np.amin(rmse)
            if best_mae(minMAE, grid_y, grid_x):
                countMAE +=1
            if best_rmse(minRMSE, grid_y, grid_x):
                countRMSE +=1

            np.savetxt('test/' + str(grid_x) + '_' + str(grid_y) + '.csv', test.Y, delimiter=',', fmt='%10.5f')
            np.savetxt('pred/' + str(grid_x) + '_' + str(grid_y) + '.csv', predictions, delimiter=',', fmt='%10.5f')
            np.savetxt('mae/' + str(grid_x) + '_' + str(grid_y) + '.csv', mae, delimiter=',', fmt='%10.5f')
            np.savetxt('rmse/' + str(grid_x) + '_' + str(grid_y) + '.csv', rmse, delimiter=',', fmt='%10.5f')

            # predictions.tofile('pred/' + grid_x + '_' + grid_y + '.csv', sep=',', format='%10.5f')
            # mae.tofile('mae/' + grid_x + '_' + grid_y + '.csv', sep=',', format='%10.5f')
            # rmse.tofile('rmse/' + grid_x + '_' + grid_y + '.csv', sep=',', format='%10.5f')

            # df1 = pd.DataFrame(predictions)
            # df1.to_csv('pred/' + grid_x + '_' + grid_y + '.csv')

            print('MAE')
            print('lin MAE:', mae[0])
            print('rf MAE:', mae[1])
            print('nnr MAE:', mae[2])
            print('svm MAE:', mae[3])
            print('knn MAE:', mae[4])

            print('RMSE')
            print('lin RMSE:', rmse[0])
            print('rf RMSE:', rmse[1])
            print('nnr RMSE:', rmse[2])
            print('svm RMSE:', rmse[3])
            print('knn RMSE:', rmse[4])

            # print('R2')
            # print('lin R2:', r2_score(test.Y, linPredict))
            # print('rf R2:', r2_score(test.Y, rfPredict))
            # print('nnr R2:', r2_score(test.Y, nnrPredict))
            # print('svm R2:', r2_score(test.Y, svmPredict))
            # print('knn R2:', r2_score(test.Y, knnPredict))

            # for r in regressors:
            #     # print(r(test))
            #     plot_result(r(test), test)
            # plot_result(predictions[4], test)

print('Better MAE percentage: ', (countMAE/total)*100, '%')
print('Better RMSE percentage: ', (countRMSE/total)*100, '%')
