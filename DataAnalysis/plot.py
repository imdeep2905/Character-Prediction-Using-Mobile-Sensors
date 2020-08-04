
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("C:/Users/jaymin/Documents/Character-Prediction-Using-Mobile-Sensors/DataAnalysis/train_data_analysis.csv")
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
wid = 0.5
plt.bar(data['label'],data['avg'])
plt.title("Average timestamp for every label")
plt.xlabel("Labels")
plt.ylabel("Timestamps")
plt.savefig("average_train.eps",format='eps')
plt.show()
