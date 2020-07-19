import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log = pd.read_csv('SavedModels/jaymin/3/log.csv',names=['index','loss','acc'])
log = log.drop(['loss'], axis = 1)
log = log.values

X = []
Y = []
offset = [0.2, 0.4, 0.6, 0.8,1]
cnt = 0
for l in log:
	X.append(l[0] + offset[cnt])
	Y.append(l[1])
	cnt = (cnt + 1) % 5
for i in range(len(X)):
    print(f"{X[i]}")
a = 'accuracy'
b = 'loss'
plt.locator_params(axis='x', nbins=len(log) /5)
plt.plot(X, Y)
plt.ylabel(a)
plt.xlabel("epochs")
plt.show()