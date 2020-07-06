import os 
import pandas as pd
from pathlib import Path

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

if __name__ == "__main__":

    for c in CLASSES:
        PATH = Path(f'C:/Users/Deep Raval/Desktop/Projects/Character-Prediction-Using-Mobile-Sensors/UnseenTest/{c}/')
        lbl = c        
        files = list()
        for f in os.listdir(PATH):
            files.append(f'C:/Users/Deep Raval/Desktop/Projects/Character-Prediction-Using-Mobile-Sensors/UnseenTest/{c}/' + str(f))
        avg = 0.
        maxi = 0
        mini = 1000
        for f in files:
            df = pd.read_csv(f)
            avg += df.shape[0]
            maxi = max(df.shape[0], maxi)
            mini = min(df.shape[0], mini)
        with open('data_analysis.csv', mode = 'a') as f:
            f.write(f"{lbl},{len(files)},{ avg / len(files)},{maxi},{mini}\n")

