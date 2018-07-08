import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_images(images, cols, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
    # fig.suptitle('Plotting Precipitation Values. Day: 2016/05/20')
    for n, (image, title) in enumerate(zip(images, titles)):
        # a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a = fig.add_subplot(6, 4, n + 1)
        plt.axis([0, len(image[0]), 0, len(image)])
        # image[image >= 0] = 0
        # image[image > 10] = 0
        x, y = image.nonzero()
        c = image[x, y]

        im = plt.scatter(y[:], x[:], c=c[:], cmap='jet', s=1)
        # if n == 10:
        #     plt.ylabel('Vertical Grid')
        # if n == 22:
        #     plt.xlabel('Horizontal Grid')
        # plt.colorbar()

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    # plt.savefig('complicated_2v7.png')


readData = pd.read_csv('complicated_2v2.csv', header=None)
imagesArr = []
for i in range(24):
    temp = pd.to_numeric(np.array(readData[i])[:]).reshape((44, 65))
    temp[temp<0] = 0
    temp[temp>50] = 50
    imagesArr.append(temp)

# imagesArr = np.round(imagesArr/np.max(imagesArr), 1)
show_images(imagesArr, 1)