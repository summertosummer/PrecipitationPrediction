import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from itertools import chain
import csv
from sklearn.cluster import KMeans
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

dfMAE = pd.read_csv('MAE.csv', header=None)
# dfMAE.replace(r'', np.NaN)

#2D Visualizaiton
def data_visualization_2d(w_data, title, visualize=True):
    if visualize:
        # w_data[w_data<5] = 0
        x, y = w_data.nonzero()
        c = w_data[x, y]
        # print(x.shape)
        # print(y.shape)
        # print(z.shape)
        # print(c.shape)

        plt.scatter(y[:], x[:], c=c[:], cmap='jet')
        plt.title(title)
        plt.colorbar()
        plt.show()

train_x = []
def create_dataset():
    for ind in range(1, len(dfMAE)):
        a = dfMAE.iloc[ind,2:26].values
        a[a==' '] = '0.0'
        df = np.round(pd.to_numeric(a), 2)
        # print(df)
        train_x.append(df)
    # train_x[train_x == np.nan] = 0
    # print(train_x)
    return np.array(train_x)

X = create_dataset()
X = np.nan_to_num(X)
data_visualization_2d(X, '')

# X[X==np.nan] = 0
# X[X==np.inf] = 0
# X[X>=10000] = 0
print(X)

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

print(labels)
print(centroids)



