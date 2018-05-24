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
#     for i in range(13):
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(10):
#             print('second: ', j)
#             # go every folder of every prediction model
#             x = []
#             for k in range(1, 25):
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
#     for i in range(13,15):
#         print('day: ', i)
#         # Verification Data: Real Data: running the loop for every day for a given second
#         for j in range(10):
#             print('second: ', j)
#             # go every folder of every prediction model
#             y = []
#             for k in range(1, 25):
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
    for i in range(15):
        print('day: ', i)
        # Verification Data: Real Data: running the loop for every day for a given second
        for j in range(10):
            print('second: ', j)
            # go every folder of every prediction model
            x = []
            for k in range(1, 24):
                print('model: ', k)
                # reading netcdf
                b = rain_models[i, j, k, 22:25, 22:25]
                # b[b > 30000] = np.nan
                rain100 = np.array(b)
                # print(rain100)
                # train[tr_count]= rain100
                # tr_count += 1

                x.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            bt = rain_models[i, j, 0, 23, 23]
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
                b = rain_models[i, j, k, 22:25, 22:25]
                # b[b > 30000] = np.nan
                rain100 = np.array(b)
                # print(rain100)
                # test[te_count] = rain100
                # te_count += 1
                y.append(list(it.chain.from_iterable(rain100)))  # flatten the list

            btt = rain_models[i, j, 0, 23, 23]
            rainRt = np.array(btt)

            test_y.append(btt)
            test_x.append(list(it.chain.from_iterable(y)))
            # te_count += 1

    print(np.shape(train_x), np.shape(train_y), np.shape(test_x), np.shape(test_y))
    # np.savetxt("Dump/nndata1" + str('a') + "by" + str('b') + ".csv", train_x, delimiter=",", fmt='%10.5f')
    # np.savetxt("Dump/nndata2" + str('a') + "by" + str('b') + ".csv", train_y, delimiter=",", fmt='%10.5f')
    # np.savetxt("Dump/nndata3" + str('a') + "by" + str('b') + ".csv", test_x, delimiter=",", fmt='%10.5f')
    # np.savetxt("Dump/nndata4" + str('a') + "by" + str('b') + ".csv", test_y, delimiter=",", fmt='%10.5f')
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = create_training_and_testing_data()

IMG_SIZE_PX_Y = 9
IMG_SIZE_PX_X = 9
SLICE_COUNT = 24

n_classes = 1
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX_Y, IMG_SIZE_PX_X, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 64]) #78336
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                # print(batch_x)
                # print(batch_y)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)