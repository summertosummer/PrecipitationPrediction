import tensorflow as tf
from netCDF4 import Dataset
import pickle
import numpy as np
import csv
import itertools as it

# def create_training_and_testing_data():
#     netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
#     rain_models = netcdf_entire_dataset.variables['summing_models']
#     time_error_rate_file = 1 #netcdf_entire_dataset.variables['time'][:]
#     models_error_rate_file = 1 #netcdf_entire_dataset.variables['models'][:]
#
#     with open('F:/dataset/rain_data/index70.csv') as csvf:
#         ind70 = csv.reader(csvf)
#         indexi70 = list(ind70)
#         index70 = indexi70[0]
#
#     train_x = []
#     train_y = []
#     tr_count = 0
#     for i in range(18):
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(10):
#             print('second: ', j)
#             # go every folder of every prediction model
#             x = []
#             for k in range(1, 24):
#                 print('model: ', k)
#                 # reading netcdf
#                 b = rain_models[i, j, k, :, :]
#                 # b[b > 30000] = np.nan
#                 rain100 = np.array(b)
#                 # print(rain100)
#                 # train[tr_count]= rain100
#                 # tr_count += 1
#
#                 x.append(list(it.chain.from_iterable(rain100)))  # flatten the list
#
#             bt = rain_models[i, j, 0, :, :]
#             rainR = np.array(bt)
#
#             train_y.append(list(it.chain.from_iterable(rainR)))
#             train_x.append(list(it.chain.from_iterable(x)))
#             # tr_count += 1
#
#     with open('F:/dataset/rain_data/index30.csv') as csvf:
#         ind30 = csv.reader(csvf)
#         indexi30 = list(ind30)
#         index30 = indexi30[0]
#
#     test_x = []
#     test_y = []
#     te_count = 0
#     for i in range(18,23):
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(10):
#             print('second: ', j)
#             # go every folder of every prediction model
#             y = []
#             for k in range(1, 24):
#                 print('model: ', k)
#                 # reading netcdf
#                 b = rain_models[i, j, k, :, :]
#                 # b[b > 30000] = np.nan
#                 rain100 = np.array(b)
#                 # print(rain100)
#                 # test[te_count] = rain100
#                 # te_count += 1
#                 y.append(list(it.chain.from_iterable(rain100)))  # flatten the list
#
#             btt = rain_models[i, j, 0, :, :]
#             rainRt = np.array(btt)
#
#             test_y.append(list(it.chain.from_iterable(rainRt)))
#             test_x.append(list(it.chain.from_iterable(y)))
#             # te_count += 1
#
#     print(np.shape(train_x), np.shape(train_y), np.shape(test_x), np.shape(test_y))
#     return train_x, train_y, test_x, test_y
#
# def create_training_and_testing_data():
#     netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
#     rain_models = netcdf_entire_dataset.variables['summing_models']
#     time_error_rate_file = 1 #netcdf_entire_dataset.variables['time'][:]
#     models_error_rate_file = 1 #netcdf_entire_dataset.variables['models'][:]
#
#     with open('F:/dataset/rain_data/index70.csv') as csvf:
#         ind70 = csv.reader(csvf)
#         indexi70 = list(ind70)
#         index70 = indexi70[0]
#
#     train_x = []
#     train_y = []
#     tr_count = 0
#     for i in range(15):
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(10):
#             print('second: ', j)
#             # go every folder of every prediction model
#             x = []
#
#             # reading netcdf
#             b = rain_models[i, j, 22:25, 22:25]
#             # b[b > 30000] = np.nan
#             rain100 = np.array(b)
#             # print(rain100)
#             # train[tr_count]= rain100
#             # tr_count += 1
#
#             x.append(list(it.chain.from_iterable(rain100)))  # flatten the list
#
#             bt = rain_models[i, j, 0, 23, 23]
#             rainR = np.array(bt)
#
#             train_y.append(rainR)
#             train_x.append(rain100)
#             # tr_count += 1
#
#     with open('F:/dataset/rain_data/index30.csv') as csvf:
#         ind30 = csv.reader(csvf)
#         indexi30 = list(ind30)
#         index30 = indexi30[0]
#
#     test_x = []
#     test_y = []
#     te_count = 0
#     for i in range(15,20):
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(10):
#             print('second: ', j)
#             # go every folder of every prediction model
#             y = []
#
#             # reading netcdf
#             b = rain_models[i, j, 22:25, 22:25]
#             # b[b > 30000] = np.nan
#             rain100 = np.array(b)
#             # print(rain100)
#             # test[te_count] = rain100
#             # te_count += 1
#             # y.append(list(it.chain.from_iterable(rain100)))  # flatten the list
#
#             btt = rain_models[i, j, 0, 23, 23]
#             rainRt = np.array(btt)
#
#             test_y.append(rainRt)
#             test_x.append(rain100)
#             # te_count += 1
#
#     # train_y = np.array(train_y)
#     # test_y = np.array(test_y)
#     print(np.shape(train_x), np.shape(train_y), np.shape(test_x), np.shape(test_y))
#     return np.round(train_x, 4), np.round(train_y, 4), np.round(test_x, 4), np.round(test_y, 4)

def create_training_and_testing_data():
    netcdf_entire_dataset = Dataset("F:/dataset/summing_dataset.nc", "r")
    rain_models = netcdf_entire_dataset.variables['summing_models']
    time_error_rate_file = 1 #netcdf_entire_dataset.variables['time'][:]
    models_error_rate_file = 1 #netcdf_entire_dataset.variables['models'][:]

    with open('F:/dataset/rain_data/index70.csv') as csvf:
        ind70 = csv.reader(csvf)
        indexi70 = list(ind70)
        index70 = indexi70[0]

    train_x = []
    train_y = []
    tr_count = 0
    for i in range(20):
        print('day: ', i)
        # Verification Data: Real Data: running the loop for every day for a given second
        for j in range(10):
            print('second: ', j)
            # go every folder of every prediction model
            x = []
            for k in range(1, 24):
                print('model: ', k)
                # reading netcdf
                b = rain_models[i, j, k, 24:27, 24:27]
                # b[b > 30000] = np.nan
                rain100 = np.array(b)
                # print(rain100)
                # train[tr_count]= rain100
                # tr_count += 1

                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, 25, 25]
            rainR = np.array(bt)

            train_y.append(rainR)
            train_x.append(list(it.chain.from_iterable(x)))
            # tr_count += 1

    with open('F:/dataset/rain_data/index30.csv') as csvf:
        ind30 = csv.reader(csvf)
        indexi30 = list(ind30)
        index30 = indexi30[0]

    test_x = []
    test_y = []
    te_count = 0
    for i in range(15,20):
        print('day: ', i)
        # Verification Data: Real Data: running the loop for every day for a given second
        for j in range(10):
            print('second: ', j)
            # go every folder of every prediction model
            y = []
            for k in range(1, 24):
                print('model: ', k)
                # reading netcdf
                b = rain_models[i, j, k, 25:27, 25:26]
                # b[b > 30000] = np.nan
                rain100 = np.array(b)
                # print(rain100)
                # test[te_count] = rain100
                # te_count += 1
                y.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            btt = rain_models[i, j, 0, 24, 26]
            rainRt = np.array(btt)

            test_y.append(btt)
            test_x.append(list(it.chain.from_iterable(y)))
            # te_count += 1

    print(np.shape(train_x), np.shape(train_y), np.shape(test_x), np.shape(test_y))
    np.savetxt("Dump/nndataTx" + str('24') + "by" + str('24') + ".csv", train_x, delimiter=",", fmt='%10.5f')
    np.savetxt("Dump/nndataTy" + str('24') + "by" + str('24') + ".csv", train_y, delimiter=",", fmt='%10.5f')
    # np.savetxt("Dump/nndata3" + str('a') + "by" + str('b') + ".csv", test_x, delimiter=",", fmt='%10.5f')
    # np.savetxt("Dump/nndata4" + str('a') + "by" + str('b') + ".csv", test_y, delimiter=",", fmt='%10.5f')
    return train_x, train_y, test_x, np.array(test_y)


train_x, train_y, test_x, test_y = create_training_and_testing_data()

n_nodes_hl1 = 500
n_nodes_hl2 = 50
n_nodes_hl3 = 5

n_classes = 1
batch_size = 50
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes]))}


# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end]).reshape((50, 1))
                # print(batch_x)
                # print(batch_y)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y.reshape((50, 1))}))


train_neural_network(x)



















#
# import tensorflow as tf
# from netCDF4 import Dataset
# import random
# import numpy as np
# import csv
#
# train_x = []
# test_x = []
#
# def create_training_and_testing_data():
#     netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
#     rain_models = netcdf_entire_dataset.variables['rain_models']
#     time_error_rate_file = 1 #netcdf_entire_dataset.variables['time'][:]
#     models_error_rate_file = 1 #netcdf_entire_dataset.variables['models'][:]
#
#     with open('F:/dataset/rain_data/index70.csv') as csvf:
#         ind70 = csv.reader(csvf)
#         indexi70 = list(ind70)
#         index70 = indexi70[0]
#
#     for i in index70:
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(time_error_rate_file):
#             print('second: ', j)
#             # go every folder of every prediction model
#             for k in range(models_error_rate_file):
#                 print('model: ', k)
#                 # reading netcdf
#                 b = rain_models[i, j, k, :, :]
#                 b[b > 10000] = np.nan
#                 rain100 = np.array(b[:100])
#                 # print(rain100)
#                 train_x.append(rain100)
#
#     with open('F:/dataset/rain_data/index30.csv') as csvf:
#         ind30 = csv.reader(csvf)
#         indexi30 = list(ind30)
#         index30 = indexi30[0]
#
#     for i in index30:
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(time_error_rate_file):
#             print('second: ', j)
#             # go every folder of every prediction model
#             for k in range(models_error_rate_file):
#                 print('model: ', k)
#                 # reading netcdf
#                 b = rain_models[i, j, k, :, :]
#                 b[b > 10000] = np.nan
#                 rain100 = np.array(b[:100])
#                 # print(rain100)
#                 test_x.append(rain100)
#
#     return train_x, test_x
#
# train_x, test_x = create_training_and_testing_data()
# n_nodes_hl = 1155*1683
# batch_size = 100
# hm_epochs = 10
#
# x = tf.placeholder('float')
# y = tf.placeholder('float')
#
#
# def neural_network_model(data):
#     hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl, n_nodes_hl])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}
#
#     hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl, n_nodes_hl])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}
#
#     hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl, n_nodes_hl])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}
#
#     output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl, n_nodes_hl])),
#                     'biases': tf.Variable(tf.random_normal([n_nodes_hl])), }
#
#     l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
#     l1 = tf.nn.relu(l1)
#
#     l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
#     l2 = tf.nn.relu(l2)
#
#     l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
#     l3 = tf.nn.relu(l3)
#
#     output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
#
#     return output
#
#
# def train_neural_network(x):
#     prediction = neural_network_model(x)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
#     optimizer = tf.train.AdamOptimizer.minimize(cost)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             i = 0
#             while i < len(train_x):
#                 start = i
#                 end = i + batch_size
#                 batch_x = np.array(train_x[start:end])
#
#                 _, c = sess.run([optimizer, cost], feed_dict={x: batch_x})
#                 epoch_loss += c
#                 i += batch_size
#
#             print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#
#         print('Accuracy:', accuracy.eval({x: test_x}))
#
#
# train_neural_network(x)
#
# '''
# def create_training_and_testing_data():
#     netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
#     rain_models = netcdf_entire_dataset.variables['rain_models']
#
#     random.shuffle(rain_models)
#     rain_models = np.array(rain_models)
#
#     testing_size = int(0.3*len(rain_models))
#
#     train_x = list(rain_models[:-testing_size, :, :])
#     test_x = list(rain_models[:-testing_size, :, :])
#
#     return train_x,test_x
# '''