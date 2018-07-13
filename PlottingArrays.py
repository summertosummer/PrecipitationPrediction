import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_images(images, cols, titles):
    min_v = np.nanmin(images)
    max_v = np.nanmax(images[images != np.inf])
    print(min_v, max_v)
    # assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
    fig.suptitle(titles)
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(6, 4, n + 1)
        plt.axis([0, len(image[0]), 0, len(image)])
        # image[image >= 0] = 0
        # image[image > 10] = 0
        x, y = image.nonzero()
        c = image[x, y]

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=1)
        if n == 8:
            plt.ylabel('Vertical Grid')
        if n == 21:
            plt.xlabel('Horizontal Grid')
        plt.clim(min_v, max_v)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    plt.savefig('array'+str(cols)+'.png')

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

def plotArray1():
    arrCSV = pd.read_csv('array1.csv', header=None)
    array1 = []
    for i in range(24):
        temp = np.array(arrCSV[:])[i]
        temp[temp == ' '] = np.nan
        temp[temp == 'nan'] = np.nan
        temp = pd.to_numeric(temp).reshape((44, 65))
        array1.append(temp)
    show_images(array1, 1, titles="Plotting the Array1 for each model")

def plotArray2():
    arrCSV = pd.read_csv('array2.csv', header=None)
    array2 = []
    for i in range(24):
        temp = np.array(arrCSV[:])[i]
        temp[temp == ' '] = np.nan
        temp[temp == 'nan'] = np.nan
        temp = pd.to_numeric(temp).reshape((44, 65))
        array2.append(temp)
    show_images(array2, 2, titles="Plotting the Array2 for each model")

# plotArray1()
# plotArray2()

arrCSV = pd.read_csv('array2.csv', header=None)
temp = np.array(arrCSV[:])[26]
temp = pd.to_numeric(temp).reshape((44, 65))
data_visualization_2dr(temp, "")