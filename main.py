import sys
sys.setrecursionlimit(1000000007)
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import LSTM,Dense,Softmax,Input,Lambda, Flatten, Reshape, Lambda, GRU, Dropout, BatchNormalization
from tensorflow.keras import Sequential
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from pathlib import Path
from io import StringIO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path("./ProcessedDataset2")

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

def process_path(file_path):
    print(type(file_path))
    print(StringIO(str(file_path.numpy(), 'utf-8')))
    return tf.io.read_file(file_path), tf.strings.split(file_path, os.sep)[-2]

def input_pipeline(test_samples, split):   
    list_ds = tf.data.Dataset.list_files(str(ROOT/'*/*.csv'))
    labeled_ds = None
    dataset_list = []
    cnt = 0
    for name in list_ds:
        try:
            label = str(name.numpy())[20]
            X = (pd.read_csv(Path(str(name.numpy(), 'utf-8'))).values)
            X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
            X = StandardScaler().fit_transform(X)
            X = np.expand_dims(X, axis = 0)
            X = np.expand_dims(X, axis = 0)
            Y = np.zeros((1, 36))
            Y[0][CLASSES.index(label)] = 1.
            Y = np.expand_dims(Y, axis = 0)
            cnt += 1
            if labeled_ds == None:
                labeled_ds = tf.data.Dataset.from_tensor_slices((X, Y)) 
            else:
                labeled_ds = labeled_ds.concatenate(tf.data.Dataset.from_tensor_slices((X, Y))) 
        except Exception as e:
            print(str(name.numpy()))
        
        if cnt % split == 0:
            dataset_list.append(labeled_ds)
            labeled_ds = None
            
        if cnt == test_samples:
            dataset_list.append(labeled_ds)
            labeled_ds = None
            test_samples = -1
            cnt = 0
            
    dataset_list.append(labeled_ds)
    return dataset_list
     
def train_model_V1(dataset, epochs):    
    model = Sequential([
        LSTM(144, return_sequences = True, input_shape = (None, 12)),
        LSTM(90, return_sequences = False),
        Dense(72, activation = "tanh"),
        Dense(36, activation = "softmax")
    ])
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    #optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, nesterov = True)
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ["accuracy"])
    for ep in range(epochs):
        for i in range(1, len(dataset)):
            model.fit(dataset[i])
        print(f'Testing after epoch no. {ep}')
        model.evaluate(dataset[0])

    print('Testing Stats FINAL:')
    model.evaluate(dataset[0])
    model.save('model.h5')
    
if __name__ == "__main__":	
    tfds_list = input_pipeline(250, 500)
    train_model_V1(tfds_list, 5)
