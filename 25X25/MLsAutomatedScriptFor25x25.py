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
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

if not os.path.exists('models25_25_9'):
    os.makedirs('models25_25_9')
if not os.path.exists('test'):
    os.makedirs('test')
if not os.path.exists('pred'):
    os.makedirs('pred')
if not os.path.exists('mae'):
    os.makedirs('mae')
if not os.path.exists('rmse'):
    os.makedirs('rmse')
if not os.path.exists('features'):
    os.makedirs('features')
if not os.path.exists('pca'):
    os.makedirs('pca')
if not os.path.exists('coef'):
    os.makedirs('coef')

#access netcdf data file
netcdf_entire_dataset = Dataset("F:/dataset/rain_data/summing_dataset.nc", "r")
rain_models = netcdf_entire_dataset.variables['summing_models']

with open('../random70.csv') as csvf:
    ind70 = csv.reader(csvf)
    indexi70 = list(ind70)
    index70 = indexi70[0]

with open('../random30.csv') as csvf:
    ind30 = csv.reader(csvf)
    indexi30 = list(ind30)
    index30 = indexi30[0]

#read MAE and RMSE files
dfMAE = pd.read_csv('MAE25x25.csv', header=None)
dfRMSE = pd.read_csv('RMSE25x25.csv', header=None)

'''
# commented out plotting area, because Dr. Hamdy's machine doesn't have pyplot installed
# ploting result
def plot_result(predictedValue, OriginalValue):
    mpl.style.use('seaborn')
    plt.plot(OriginalValue.X, 'o')
    plt.plot(OriginalValue.Y, color='Green', label='Actual')
    plt.plot(predictedValue, color='Red', label='New Prediction')
    plt.legend()
    plt.show()

#plotting input
def plot_input(x, y):
    mpl.style.use('seaborn')
    plt.plot(x, 'o')
    plt.plot(y, color='red', label='Real')
    plt.legend()
    plt.show()
'''

# creating the whole dataset (inputs and target), not dividing into training and testing here
def create_training_data(grid_x, grid_y):
    data_x = [] # inputs
    data_y = [] # target
    tr_count = 0
    for i in index70: # working with 20 days
        for j in range(10): # 10 times in each day
            x = []
            for k in range(1, 25): # 24 models as input
                # print('model: ', k)
                b = rain_models[i, j, k, grid_y - 1:grid_y + 2, grid_x - 1:grid_x + 2] #taking an area of 9X9 from every model
                rain100 = np.array(b)
                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, grid_y, grid_x] #taking the real data as target, zero in the third dimention is for real data
            rainR = bt

            data_y.append(rainR) # appending real rain data
            data_x.append(list(it.chain.from_iterable(x))) # appending inputs

    return data_x, data_y

# creating the whole dataset (inputs and target), not dividing into training and testing here
def create_testing_data(grid_x, grid_y):
    data_x = [] # inputs
    data_y = [] # target
    tr_count = 0
    for i in index30: # working with 20 days
        for j in range(10): # 10 times in each day
            x = []
            for k in range(1, 25): # 24 models as input
                # print('model: ', k)
                b = rain_models[i, j, k, grid_y - 1:grid_y + 2, grid_x - 1:grid_x + 2] #taking an area of 9X9 from every model
                rain100 = np.array(b)
                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, grid_y, grid_x] #taking the real data as target, zero in the third dimention is for real data
            rainR = bt

            data_y.append(rainR) # appending real rain data
            data_x.append(list(it.chain.from_iterable(x))) # appending inputs

    return data_x, data_y

# dividing into training and testing dataset here
# training new models using the training data
# storing the new models
# predicting target for the testing data
# calculating mae and rmse of the new prediction
def run_models(grid_y, grid_x):
    X_train, Y_train = create_training_data(grid_x, grid_y) # X and Y is the inputs and target
    data = Table(X_train, Y_train) # creating a Orange table combining both X and Y

    # print(data.domain)

    # print(data.Y)
    # np.savetxt('data/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(data), delimiter=',', fmt='%10.5f')
    # print(out_data.domain)
    # print(out_data.Y)
    # print(data.domain)

    feature_method = og.preprocess.score.UnivariateLinearRegression() # feature selection
    selector = og.preprocess.SelectBestFeatures(method=feature_method, k=50) # taking 50 features out of 216
    out_data2 = selector(data) # this is the new dataset with 50 features
    np.savetxt('features/' + str(grid_x) + '_' + str(grid_y) + '.csv', out_data2, delimiter=',', fmt='%10.5f')
    # plot_input(out_data2.X, out_data2.Y)
    # print(out_data2.domain)
    # print(out_data2)

    pca = PCA(n_components=5) # PCA with 5 components
    model = pca(out_data2)
    train = model(out_data2)
    # print(pca.domain)
    # print(model.components_)
    temp = []
    temp.append(pca.domain)
    for arr in model.components_:
        temp.append(list(arr))
    # temp.append(model.components_)
    np.savetxt('pca/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(temp), delimiter=',', fmt='%s')
    # print(out_data.domain)

    ############################################
    # dividing into training and testing dataset
    # test = og.data.Table(out_data.domain, random.sample(out_data, 60))
    # train = og.data.Table(out_data.domain, [d for d in out_data if d not in test])
    ############################################

    X_test, Y_test = create_testing_data(grid_x, grid_y)  # X and Y is the inputs and target
    data2 = Table(X_test, Y_test)  # creating a Orange table combining both X and Y

    # print(data.Y)
    # np.savetxt('data/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(data), delimiter=',', fmt='%10.5f')
    # print(out_data.domain)
    # print(out_data.Y)

    feature_method2 = og.preprocess.score.UnivariateLinearRegression()  # feature selection
    selector2 = og.preprocess.SelectBestFeatures(method=feature_method2, k=50)  # taking 50 features out of 216
    out_data22 = selector2(data2)  # this is the new dataset with 50 features
    # plot_input(out_data2.X, out_data2.Y)
    # print(out_data2.domain)

    pca2 = PCA(n_components=5)  # PCA with 5 components
    model2 = pca2(out_data22)
    test = model2(out_data22)
    # print(out_data.domain)

    # ML models
    lin = LinearRegression()
    # lin = og.regression.linear.LinearRegressionLearner()
    rf = og.regression.random_forest.RandomForestRegressionLearner()
    nnr = og.regression.NNRegressionLearner()
    svm = og.regression.SVRLearner()
    knn = KNeighborsRegressor(n_neighbors=7) #knn from sci-kit learn, rest of them are from Orange

    # fitting data into ML models
    learners = [rf, nnr, svm]
    regressors = [learner(train) for learner in learners]
    lin.fit(train.X, train.Y)
    knn.fit(train.X, train.Y)
    # print(lin.coef_)
    np.savetxt('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', lin.coef_, delimiter=',', fmt='%10.5f')
    # np.savetxt('coef/' + str(grid_x) + '_' + str(grid_y) + '.csv', knn.coef_, delimiter=',', fmt='%10.5f')

    #saving the new trained models
    with open("models25_25_9/"+str(grid_x)+"_"+str(grid_y)+"_lin.pickle", "wb") as f:
        pickle.dump(lin, f)
    with open("models25_25_9/"+str(grid_x)+"_"+str(grid_y)+"_rf.pickle", "wb") as f:
        pickle.dump(rf, f)
    with open("models25_25_9/"+str(grid_x)+"_"+str(grid_y)+"_nnr.pickle", "wb") as f:
        pickle.dump(nnr, f)
    with open("models25_25_9/"+str(grid_x)+"_"+str(grid_y)+"_svm.pickle", "wb") as f:
        pickle.dump(svm, f)
    with open("models25_25_9/"+str(grid_x)+"_"+str(grid_y)+"_knn.pickle", "wb") as f:
        pickle.dump(knn, f)

    # predicting target for testing dataset
    # print((r(test)[0] for r in regressors))
    linPredict = lin.predict(test.X)
    # linPredict = regressors[0](test)
    rfPredict = regressors[0](test)
    nnrPredict = regressors[1](test)
    svmPredict = regressors[2](test)
    knnPredict = knn.predict(test.X)

    # storing the predictions into array
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

    # calculating MAE and RMSE of the new prediction and storing them into arrays
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

    return np.array(mae), np.array(rmse), np.array(predictions), test # returning error rates, predictions


# finding if the new MAE is better then the old existing models' MAE or not
# existing_model_value1 = 0
def best_mae(minMAE, grid_y, grid_x):
    global existing_model_value1
    # print('min MAE:', minMAE)
    # taking the row where for a particular grid
    df1 = dfMAE[(dfMAE[0] == str(grid_y)) & (dfMAE[1] == str(' ') + str(grid_x))]

    flag = True
    arr = []
    existing_model_value1 = 999999 # considering a high value
    for z in range(2, 26): # for each models
        value = pd.to_numeric(dfMAE[z][df1.index], errors='coerce')
        # print(value.values[0])
        if value.values[0] < existing_model_value1:
            existing_model_value1 = value.values[0] # taking the lowest MAE
        if value.values[0] < minMAE: arr.append(1)
        else: arr.append(0)

    if existing_model_value1 < minMAE:
        flag = False

    return flag, arr

# finding if the new RMSE is better then the old existing models' RMSE or not
# existing_model_value2 = 0
def best_rmse(minRMSE, grid_y, grid_x):
    global existing_model_value2
    # print('min rmse:', minRMSE)
    # taking the row where for a particular grid
    df1 = dfRMSE[(dfRMSE[0] == str(grid_y)) & (dfRMSE[1] == str(' ') + str(grid_x))]

    flag = True
    arr = []
    existing_model_value2 = 999999
    for z in range(2, 26): # for each models
        value = pd.to_numeric(dfRMSE[z][df1.index], errors='coerce')
        # print(value.values[0])
        if value.values[0] < existing_model_value2:
            existing_model_value2 = value.values[0] #taking the lowest RMSE
        if value.values[0] < minRMSE: arr.append(1)
        else: arr.append(0)

    if existing_model_value2 < minRMSE:
        flag = False

    return flag, arr


# saving the model info into file
check = open('ModelsInfo25x25_v2.csv', 'w')
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
check.write(str('Old RMSE better in'))
check.write(', ')
check.write(str('Old MAE Better in'))
check.write(', ')
check.write(str('Old RMSE and MAE Better in'))
check.write(', ')
check.write(str('Cumulative Accuracy RMSE'))
check.write(', ')
check.write(str('Cumulative Accuracy MAE'))
check.write('\n')

total = 0
countMAE = 0
countRMSE = 0
for grid_y in range(22, 45): # for every y
    for grid_x in range(22, 66): # for every x
        print('=================PLACE:', grid_x, grid_y, '=====================')

        flag = True
        for _ in range(1): # looping 15 times to find the best model
            # try:
                mae, rmse, predictions, test = run_models(grid_y, grid_x)
                minRMSE = np.amin(rmse) # minimum RMSE from the new models
                # total += 1
                getFlag, _ = best_rmse(minRMSE, grid_y, grid_x)
                if getFlag: # checking if it is the best one
                    print('found the best')
                    break
            # except:
            #     flag = False
            #     pass

        # print("Learner  RMSE  MAE  R2")
        # for i in range(len(learners)):
        #     print("{:8s} {:.2f} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], mae[i], r2[i]))

        if flag:
            total += 1
            minMAE = np.amin(mae) # minimum MAE of new models
            minRMSE = np.amin(rmse) # minimum RMSE of new models
            getFlagMAE, arrMAE = best_mae(minMAE, grid_y, grid_x)
            getFlagRMSE, arrRMSE = best_rmse(minRMSE, grid_y, grid_x)
            if getFlagMAE:
                countMAE += 1
            if getFlagRMSE:
                countRMSE += 1

            # saving mae, rmse, prediction and target data
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

            # finding the name of the best model
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
            check.write(str(sum(arrRMSE)))
            check.write(', ')
            check.write(str(sum(arrMAE)))
            check.write(', ')
            check.write(str(sum([i == j and i == 1 for i, j in zip(arrRMSE, arrMAE)])))
            check.write(', ')
            check.write(str(per_rmse))
            check.write(', ')
            check.write(str(per_mae))
            check.write('\n')

            print('MAE is better in', countMAE, '/', total, '=', per_mae, '% cases')
            print('RMSE is better in', countRMSE, '/', total, '=', per_rmse, '% cases')
        else:
            check.write(str(grid_y))
            check.write(', ')
            check.write(str(grid_x))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write(', ')
            check.write(str(0))
            check.write('\n')
