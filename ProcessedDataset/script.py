import os 
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    cnt = 0
    lbl = input('Enter Label : ')        
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
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
    for f in files:
        if 'csv' in f:
            df = pd.read_csv(f)
            df = df.filter(take, axis=1)
            df.to_csv(f'{lbl}_{cnt}.csv', index = False)
            print(f'Handling {f}')
            os.remove(f)
            cnt += 1