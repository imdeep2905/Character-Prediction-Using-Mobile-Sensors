import os 
import pandas as pd
from pathlib import Path

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

if __name__ == "__main__":
    for c in CLASSES:
        PATH = Path(f'C:/Users/Deep Raval/Desktop/Projects/Character-Prediction-Using-Mobile-Sensors/ProcessedDataset3/{c}/')
        cnt = 0
        lbl = c        
        files = list()
        for f in os.listdir(PATH):
            files.append(f'C:/Users/Deep Raval/Desktop/Projects/Character-Prediction-Using-Mobile-Sensors/ProcessedDataset3/{c}/' + str(f))
        take = ['ACCELEROMETER X (m/s²)',
                'ACCELEROMETER Y (m/s²)',
                'ACCELEROMETER Z (m/s²)',
                'LINEAR ACCELERATION X (m/s²)',
                'LINEAR ACCELERATION Y (m/s²)',
                'LINEAR ACCELERATION Z (m/s²)',
                'GYROSCOPE X (rad/s)',
                'GYROSCOPE Y (rad/s)',
                'GYROSCOPE Z (rad/s)',
                'MAGNETIC FIELD X (μT)',
                'MAGNETIC FIELD Y (μT)',
                'MAGNETIC FIELD Z (μT)']
        currpted = list()
        for file in files:
            try:
                if 'csv' in f:
                    df = pd.read_csv(file)
                    df = df.filter(take, axis=1)
                    df.to_csv(os.path.join(PATH, f'{lbl}__{cnt}.csv'), index = False)
                    print(f'Handling {file}')
                    print('     Saving it as : ', f'{lbl}__{cnt}.csv')
                    os.remove(file)
                    cnt += 1
            except Exception:
                currpted.append(file)
    print("----------------------------------------------------------------------\n\n")
    print(currpted)
    print('TOTAL FILE PROCESSED : ', cnt)