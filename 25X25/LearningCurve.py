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
def run_models(X_train, Y_train, X_test, Y_test, grid_y, grid_x, intvl):
    # X_train, Y_train = create_training_data(grid_x, grid_y) # X and Y is the inputs and target

    # for intvl in range(1, 140):
    data = Table(X_train[0:intvl], Y_train[0:intvl]) # creating a Orange table combining both X and Y
    # print(data)

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
    train2 = model(out_data2)
    # print(model.singular_values_, '----------------------------')
    # print(model.explained_variance_ )
    # print(model.explained_variance_ratio_)
    # print(pca.domain)
    # print(model.components_)

    featuresIndex = set()

    for comp in range(len(model.components_)-1, 0, -1):
        top2 = (- np.array(model.components_[comp])).argsort()[:2]
        featuresIndex |= set(top2)

    top2 = (- np.array(model.components_[0])).argsort()[:13]
    f_index = 0
    while(len(featuresIndex) != 13):
        featuresIndex.add(top2[f_index])
        f_index += 1

    ind = np.array(list(featuresIndex))
    # print(ind)
    # print(out_data2[:,ind])

    train = Table(list(out_data2[:,ind]), Y_train[0:intvl])
    # print(train)



    # temp = []
    # temp.append(pca.domain)
    # for arr in model.components_:
    #     temp.append(list(arr))
    # # temp.append(model.components_)
    # np.savetxt('pca/' + str(grid_x) + '_' + str(grid_y) + '.csv', np.array(temp), delimiter=',', fmt='%s')
    # # print(out_data.domain)

    ############################################
    # dividing into training and testing dataset
    # test = og.data.Table(out_data.domain, random.sample(out_data, 60))
    # train = og.data.Table(out_data.domain, [d for d in out_data if d not in test])
    ############################################

    # X_test, Y_test = create_testing_data(grid_x, grid_y)  # X and Y is the inputs and target
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
    # test = model2(out_data22)
    test = Table(list(out_data22[:,ind]), Y_test)
    # print(out_data.domain)

    # ML models
    lin = LinearRegression()
    # lin = og.regression.linear.LinearRegressionLearner()
    rf = og.regression.random_forest.RandomForestRegressionLearner()
    nnr = og.regression.NNRegressionLearner()
    svm = og.regression.SVRLearner()
    knn = KNeighborsRegressor(n_neighbors=7) #knn from sci-kit learn, rest of them are from Orange
    poly = og.regression.linear.PolynomialLearner(degree=2)

    # fitting data into ML models
    learners = [rf, nnr, svm, poly]
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
    with open("models25_25_9/"+str(grid_x)+"_"+str(grid_y)+"_poly.pickle", "wb") as f:
        pickle.dump(poly, f)

    # predicting target for testing dataset
    # print((r(test)[0] for r in regressors))
    linPredict = lin.predict(test.X)
    # linPredict = regressors[0](test)
    rfPredict = regressors[0](test)
    nnrPredict = regressors[1](test)
    svmPredict = regressors[2](test)
    knnPredict = knn.predict(test.X)
    polyPredict = regressors[3](test)

    # storing the predictions into array
    predictions = []
    predictions.append(linPredict)
    predictions.append(rfPredict)
    predictions.append(nnrPredict)
    predictions.append(svmPredict)
    predictions.append(knnPredict)
    predictions.append(polyPredict)


    # training errors
    train_linPredict = lin.predict(train.X)
    # linPredict = regressors[0](test)
    train_rfPredict = regressors[0](train)
    train_nnrPredict = regressors[1](train)
    train_svmPredict = regressors[2](train)
    train_knnPredict = knn.predict(train.X)
    train_polyPredict = regressors[3](train)

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
    rmse.append(math.sqrt(mean_squared_error(test.Y, polyPredict)))

    # mae.append(mean_absolute_error(test.Y, linPredict))
    mae.append(mean_absolute_error(test.Y, rfPredict))
    # mae.append(mean_absolute_error(test.Y, nnrPredict))
    # mae.append(mean_absolute_error(test.Y, svmPredict))
    # mae.append(mean_absolute_error(test.Y, knnPredict))
    # mae.append(mean_absolute_error(test.Y, polyPredict))

    train_rmse = []
    train_rmse.append(math.sqrt(mean_squared_error(train.Y, train_linPredict)))
    train_rmse.append(math.sqrt(mean_squared_error(train.Y, train_rfPredict)))
    train_rmse.append(math.sqrt(mean_squared_error(train.Y, train_nnrPredict)))
    train_rmse.append(math.sqrt(mean_squared_error(train.Y, train_svmPredict)))
    train_rmse.append(math.sqrt(mean_squared_error(train.Y, train_knnPredict)))
    train_rmse.append(math.sqrt(mean_squared_error(train.Y, train_polyPredict)))

    train_mae = []
    # train_mae.append(mean_absolute_error(train.Y, train_linPredict))
    train_mae.append(mean_absolute_error(train.Y, train_rfPredict))
    # train_mae.append(mean_absolute_error(train.Y, train_nnrPredict))
    # train_mae.append(mean_absolute_error(train.Y, train_svmPredict))
    # train_mae.append(mean_absolute_error(train.Y, train_knnPredict))
    # train_mae.append(mean_absolute_error(train.Y, train_polyPredict))

    return np.array(mae), np.array(rmse), np.array(train_rmse), np.array(train_mae), np.array(predictions), test # returning error rates, predictions


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

total = 0
countMAE = 0
countRMSE = 0
tempCheck = []
for grid_y in range(3, 4): # for every y //45
    for grid_x in range(50, 51): # for every x// 66
        print('=================PLACE:', grid_x, grid_y, '=====================')
        tempCheck = rain_models[:20, :10, 0, grid_y, grid_x]
        if not tempCheck.any():
            continue


########################## LEARNING CURVE CODE STARTS HERE ##########################
        flag = True
        learning_curve = []
        X_train, Y_train = create_training_data(grid_x, grid_y)  # X and Y is the inputs and target
        X_test, Y_test = create_testing_data(grid_x, grid_y)
        for indx in range(10, 140): # looping 15 times to find the best model
            # try:
            mae, rmse, t_rmse, t_mae, predictions, test = run_models(X_train, Y_train, X_test, Y_test, grid_y, grid_x, indx)
            minRMSE = np.amin(rmse) # minimum RMSE from the new models
            minTRMSE = np.amin(t_rmse)

            minMAE = np.amin(mae)
            minTMAE = np.amin(t_mae)

            learning_curve.append([indx, minTMAE, minMAE])
            # print(indx, minTRMSE, minRMSE)
            print(indx, minTMAE, minMAE)

        np.savetxt('learning_curve' + str(grid_x) + '_' + str(grid_y) + '.csv', learning_curve, delimiter=',', fmt='%10.5f')

########################## LEARNING CURVE CODE ENDS HERE ##########################


            # total += 1
            # getFlag, _ = best_rmse(minRMSE, grid_y, grid_x)
            # if getFlag: # checking if it is the best one
            #     print('found the best')
            #     break

            # if flag:
            #     total += 1
            #     minMAE = np.amin(mae) # minimum MAE of new models
            #     minRMSE = np.amin(rmse) # minimum RMSE of new models
            #     getFlagMAE, arrMAE = best_mae(minMAE, grid_y, grid_x)
            #     getFlagRMSE, arrRMSE = best_rmse(minRMSE, grid_y, grid_x)
            #     if getFlagMAE:
            #         countMAE += 1
            #     if getFlagRMSE:
            #         countRMSE += 1


