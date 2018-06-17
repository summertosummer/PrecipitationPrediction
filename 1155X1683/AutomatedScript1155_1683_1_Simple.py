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
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os

if not os.path.exists('models1155_1683_1s'):
    os.makedirs('models1155_1683_1s')
if not os.path.exists('test1155_1683_1s'):
    os.makedirs('test1155_1683_1s')
if not os.path.exists('pred1155_1683_1s'):
    os.makedirs('pred1155_1683_1s')
if not os.path.exists('mae1155_1683_1s'):
    os.makedirs('mae1155_1683_1s')
if not os.path.exists('rmse1155_1683_1s'):
    os.makedirs('rmse1155_1683_1s')

#access netcdf data file
netcdf_entire_dataset = Dataset("entire_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['rain_models']

#read MAE and RMSE files
dfMAE = pd.read_csv('MAE1155_1683.csv', header=None)
dfRMSE = pd.read_csv('RMSE1155_1683.csv', header=None)

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
            for k in range(1, 25):
                # print('model: ', k)
                b = rain_models[i, j, k, grid_y, grid_x]
                # rain100 = np.array(b)
                x.append(b)  # flatten the list

            bt = rain_models[i, j, 0, grid_y, grid_x]
            rainR = bt

            data_y.append(rainR)
            data_x.append(x)

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    data_x[data_x == np.inf] = np.nan
    data_y[data_y == np.inf] = np.nan

    col_mean = np.nanmean(data_x, axis=0)
    # Find indicies that you need to replace
    inds = np.where(np.isnan(data_x))
    # Place column means in the indices. Align the arrays using take
    data_x[inds] = np.take(col_mean, inds[1])

    col_mean2 = np.nanmean(data_y, axis=0)
    # Find indicies that you need to replace
    inds2 = np.where(np.isnan(data_y))
    # Place column means in the indices. Align the arrays using take
    data_y[inds2] = np.take(col_mean2, inds2[0])

    return data_x, data_y


def run_models(grid_y, grid_x):
    X, Y = create_training_and_testing_data(grid_x, grid_y)
    data = Table(X, Y)
    # print(data.Y)
    # np.savetxt('data/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(data), delimiter=',', fmt='%10.5f')
    # print(out_data.domain)
    # print(out_data.Y)

    # feature_method = og.preprocess.score.UnivariateLinearRegression()
    # selector = og.preprocess.SelectBestFeatures(method=feature_method, k=10)
    # out_data2 = selector(data)
    # plot_input(out_data2.X, out_data2.Y)
    # print(out_data2.domain)

    # pca = PCA(n_components=5)
    # model = pca(out_data2)
    # out_data = model(out_data2)
    # print(out_data.domain)

    test = og.data.Table(data.domain, random.sample(data, 60))
    train = og.data.Table(data.domain, [d for d in data if d not in test])

    lin = og.regression.linear.LinearRegressionLearner()
    rf = og.regression.random_forest.RandomForestRegressionLearner()
    nnr = og.regression.NNRegressionLearner()
    svm = og.regression.SVRLearner()
    knn = KNeighborsRegressor(n_neighbors=3)

    learners = [lin, rf, nnr, svm]
    regressors = [learner(train) for learner in learners]
    knn.fit(train.X, train.Y)

    with open("models1155_1683_1s/" + str(grid_x) + "_" + str(grid_y) + "_lin.pickle", "wb") as f:
        pickle.dump(lin, f)
    with open("models1155_1683_1s/" + str(grid_x) + "_" + str(grid_y) + "_rf.pickle", "wb") as f:
        pickle.dump(rf, f)
    with open("models1155_1683_1s/" + str(grid_x) + "_" + str(grid_y) + "_nnr.pickle", "wb") as f:
        pickle.dump(nnr, f)
    with open("models1155_1683_1s/" + str(grid_x) + "_" + str(grid_y) + "_svm.pickle", "wb") as f:
        pickle.dump(svm, f)
    with open("models1155_1683_1s/" + str(grid_x) + "_" + str(grid_y) + "_knn.pickle", "wb") as f:
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


# existing_model_value1 = 0
def best_mae(minMAE, grid_y, grid_x):
    global existing_model_value1
    # print('min MAE:', minMAE)
    df1 = dfMAE[(dfMAE[0] == str(grid_y)) & (dfMAE[1] == str(' ') + str(grid_x))]

    flag = True
    for z in range(2, 26):
        value = pd.to_numeric(dfMAE[z][df1.index], errors='coerce')
        # print(value.values[0])
        existing_model_value1 = value.values[0]

    if existing_model_value1 < minMAE:
        flag = False

    return flag


# existing_model_value2 = 0
def best_rmse(minRMSE, grid_y, grid_x):
    global existing_model_value2
    # print('min rmse:', minRMSE)
    df1 = dfRMSE[(dfRMSE[0] == str(grid_y)) & (dfRMSE[1] == str(' ') + str(grid_x))]

    flag = True
    for z in range(2, 26):
        value = pd.to_numeric(dfRMSE[z][df1.index], errors='coerce')
        # print(value.values[0])
        existing_model_value2 = value.values[0]

    if existing_model_value2 < minRMSE:
        flag = False

    return flag


check = open('ModelsInfo_1155_1683_1s.csv', 'w')
check.truncate()
check.write(str('Y'))
check.write(', ')
check.write(str('X'))
check.write(', ')
check.write(str('Best RMSE Model Name'))
check.write(', ')
check.write(str('Best MAE Model Name'))
check.write(', ')
check.write(str('New Model Best RMSE Value'))
check.write(', ')
check.write(str('New Model Best MAE Value'))
check.write(', ')
check.write(str('Old Model Best RMSE Value'))
check.write(', ')
check.write(str('Old Model Best MAE Value'))
check.write(', ')
check.write(str('Cumulative Accuracy RMSE'))
check.write(', ')
check.write(str('Cumulative Accuracy MAE'))
check.write('\n')

total = 0
countMAE = 0
countRMSE = 0
for grid_y in range(0, 1150):
    for grid_x in range(0, 1683):
        print('=================PLACE:', grid_x, grid_y, '=====================')

        flag = True
        for _ in range(15):
            try:
                mae, rmse, predictions, test = run_models(grid_y, grid_x)
                minRMSE = np.amin(rmse)
                # total += 1
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
            total += 1
            minMAE = np.amin(mae)
            minRMSE = np.amin(rmse)
            if best_mae(minMAE, grid_y, grid_x):
                countMAE += 1
            if best_rmse(minRMSE, grid_y, grid_x):
                countRMSE += 1

            np.savetxt('test_1155_1683_1s/' + str(grid_x) + '_' + str(grid_y) + '.csv', test.Y, delimiter=',', fmt='%10.5f')
            np.savetxt('pred_1155_1683_1s/' + str(grid_x) + '_' + str(grid_y) + '.csv', predictions, delimiter=',', fmt='%10.5f')
            np.savetxt('mae_1155_1683_1s/' + str(grid_x) + '_' + str(grid_y) + '.csv', mae, delimiter=',', fmt='%10.5f')
            np.savetxt('rmse_1155_1683_1s/' + str(grid_x) + '_' + str(grid_y) + '.csv', rmse, delimiter=',', fmt='%10.5f')

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

            best_rmse_name = ''
            best_ind_rmse = np.argmin(rmse)
            if best_ind_rmse == 0:
                best_rmse_name = 'LR'
            if best_ind_rmse == 1:
                best_rmse_name = 'RF'
            if best_ind_rmse == 2:
                best_rmse_name = 'NN'
            if best_ind_rmse == 3:
                best_rmse_name = 'SVR'
            if best_ind_rmse == 4:
                best_rmse_name = 'KNN'

            best_mae_name = ''
            best_ind_mae = np.argmin(mae)
            if best_ind_mae == 0:
                best_mae_name = 'LR'
            if best_ind_mae == 1:
                best_mae_name = 'RF'
            if best_ind_mae == 2:
                best_mae_name = 'NN'
            if best_ind_mae == 3:
                best_mae_name = 'SVR'
            if best_ind_mae == 4:
                best_mae_name = 'KNN'

            per_rmse = (countRMSE / total) * 100
            per_mae = (countMAE / total) * 100

            check.write(str(grid_y))
            check.write(', ')
            check.write(str(grid_x))
            check.write(', ')
            check.write(str(best_rmse_name))
            check.write(', ')
            check.write(str(best_mae_name))
            check.write(', ')
            check.write(str(minRMSE))
            check.write(', ')
            check.write(str(minMAE))
            check.write(', ')
            check.write(str(existing_model_value2))
            check.write(', ')
            check.write(str(existing_model_value1))
            check.write(', ')
            check.write(str(per_rmse))
            check.write(', ')
            check.write(str(per_mae))
            check.write('\n')

            print('MAE is better in', countMAE, '/', total, '=', per_mae, '% cases')
            print('RMSE is better in', countRMSE, '/', total, '=', per_rmse, '% cases')
