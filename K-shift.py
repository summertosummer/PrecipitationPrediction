import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from itertools import chain
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

# centers = [[1,1,1],[5,5,5],[3,10,10], [3,1,4]]
#
# X, _ = make_blobs(n_samples = 100, centers = centers, cluster_std = 1.5)
# print(X)

train_x = []
def create_training_and_testing_data():
    netcdf_entire_dataset = Dataset("F:/dataset/entire_dataset.nc", "r")
    rain_models = netcdf_entire_dataset.variables['rain_models']
    time_error_rate_file = 1 #netcdf_entire_dataset.variables['time'][:]
    models_error_rate_file = 1 #netcdf_entire_dataset.variables['models'][:]

    with open('F:/dataset/rain_data/index70.csv') as csvf:
        ind70 = csv.reader(csvf)
        indexi70 = list(ind70)
        index70 = indexi70[0]

    for i in index70:
        print('day: ', i)
        # Verification Data: Real Data: running the loop for every day for a given second
        for j in range(time_error_rate_file):
            print('second: ', j)
            # go every folder of every prediction model
            for k in range(models_error_rate_file):
                print('model: ', k)
                # reading netcdf
                b = rain_models[i, j, k, :, :]
                b[b > 1000] = 0
                rain100 = np.array(b[:100,:100])
                # print(rain100)
                x = list(chain.from_iterable(rain100))
                # print(x)
                train_x.append(x)
    return train_x

X = create_training_and_testing_data()
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()