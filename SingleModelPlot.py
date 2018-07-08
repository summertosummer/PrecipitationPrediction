import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

readData = pd.read_csv('complicated_2v2.csv', header=None)

#2D Visualizaiton
def data_visualization_2dr(w_data, title, i=0, visualize=True):
    if visualize:
        plt.axis([0, len(w_data[0]), 0, len(w_data)])
        # w_data[w_data >= 0] = 0
        # w_data[w_data >= 100] = 0
        x, y = w_data.nonzero()
        # x = range(0, 65)
        # y = range(0, 44)
        c = w_data[x, y]
        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        plt.title(title)
        plt.colorbar()
        # plt.savefig('com/fig' + str(i) + '.png')
        # plt.clim(-5, 0)
        plt.show()
        plt.close()

def function1(model_num, x_start, x_end, y_start, y_end):
    temp = pd.to_numeric(np.array(readData[model_num])[:]).reshape((44, 65))
    # temp[temp < 0] = 0 # commented out these two lines, might be necessary sometimes later
    # temp[temp > 50] = 50
    spec_arr = temp[y_start:y_end, x_start:x_end]

    return spec_arr

def function2(arr):
    data_visualization_2dr(arr, "numpy array plot")

numpy_arr = function1(1, 0, 30, 0, 30)
function2(numpy_arr)