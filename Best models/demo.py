
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log = pd.read_csv('logL.csv',names=['index','loss','acc'])
log = log.drop(['loss'], axis = 1)
log = log.values

log1 = pd.read_csv('logSL.csv',names=['index','loss','acc'])
log1 = log1.drop(['loss'], axis = 1)
log1 = log1.values

log2 = pd.read_csv('logB.csv',names=['index','loss','acc'])
log2 = log2.drop(['loss'], axis = 1)
log2 = log2.values

log3 = pd.read_csv('logG.csv',names=['index','loss','acc'])
log3 = log3.drop(['loss'], axis = 1)
log3 = log3.values

log4 = pd.read_csv('logCL.csv',names=['index','loss','acc'])
log4 = log4.drop(['loss'], axis = 1)
log4 = log4.values

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
X1 = []
Y1= []
for l in log1:
	X1.append(l[0] + offset[cnt])
	Y1.append(l[1])
	cnt = (cnt + 1) % 5
for i in range(len(X1)):
    print(f"{X1[i]}")


X2 = []
Y2 = []
for l in log2:
	X2.append(l[0] + offset[cnt])
	Y2.append(l[1])
	cnt = (cnt + 1) % 5
for i in range(len(X2)):
    print(f"{X2[i]}")

X3 = []
Y3 = []
for l in log3:
	X3.append(l[0] + offset[cnt])
	Y3.append(l[1])
	cnt = (cnt + 1) % 5
for i in range(len(X3)):
    print(f"{X3[i]}")


X4 = []
Y4 = []
for l in log4:
	X4.append(l[0] + offset[cnt])
	Y4.append(l[1])
	cnt = (cnt + 1) % 5
for i in range(len(X4)):
    print(f"{X4[i]}")

a = 'accuracy'
b = 'loss'
plt.locator_params(axis='x', nbins=len(log) /5)
plt.plot(X, Y,label = "LSTM")
plt.plot(X1,Y1,label = "Stateful-LSTM")
plt.plot(X2,Y2,label = "Bidirectional-LSTM")
plt.plot(X3,Y3 ,label = "GRU")
plt.plot(X4,Y4,label = "ConvLSTM")
plt.legend(loc="bottom right")
plt.ylabel(a)
plt.xlabel("epoch")
plt.show()
