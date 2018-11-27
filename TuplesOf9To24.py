import pandas as pd
import csv
import numpy as np

#read MAE and RMSE files
# ifLR = pd.read_csv("1650.csv", header=None)
# checkIfLR = np.array(ifLR[:])[13]
# print(checkIfLR)

with open('1650.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0
    for row in csv_reader:
        if count == 26:
            for r in row:
                print(r[1:-1])

        count += 1