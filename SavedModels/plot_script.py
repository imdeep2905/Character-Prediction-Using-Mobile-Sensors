import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log = pd.read_csv('log.csv',names=['index','loss','acc'])
log = log.drop(['loss'], axis = 1)
log = log.values

X = []
Y = []
offset = [0.0, 0.2, 0.4, 0.6, 0.8]
cnt = 0
for l in log:
	X.append(l[0] + offset[cnt])
	Y.append(l[1])
	cnt = (cnt + 1) % 5

plt.locator_params(axis='x', nbins=len(log) /5)
plt.plot(X, Y)
plt.show()

