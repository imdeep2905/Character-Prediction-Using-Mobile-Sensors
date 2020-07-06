
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("C:/Users/jaymin/Documents/Character-Prediction-Using-Mobile-Sensors/DataAnalysis/test_data_analysis.csv")
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
wid = 0.5
plt.bar(data['label'],data['min'])
plt.title("Minimum timestamp for every labels")
plt.xlabel("Labels")
plt.ylabel("Timestamp")
plt.show()
