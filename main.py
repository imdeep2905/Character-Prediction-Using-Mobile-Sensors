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


ROOT = Path("./ProcessedDataset2")

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

def use_gpu(gpu):
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def input_pipeline(test_samples, split):   
    list_ds = tf.data.Dataset.list_files(str(ROOT/'*/*.csv'))
    labeled_ds = None
    dataset_list = []
    cf = []
    cnt = 0
    tsdone = False
    for name in list_ds:
        try:
            label = str(name.numpy())[str(name.numpy()).find('\\') + 2]
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
            cf.append(str(name.numpy(), 'utf-8'))
        
        if cnt % split == 0:
            dataset_list.append(labeled_ds)
            labeled_ds = None
            
        if cnt == test_samples and not tsdone:
            dataset_list.append(labeled_ds)
            labeled_ds = None
            cnt = 0
            tsdone = True
            
    dataset_list.append(labeled_ds)
    print('------------------------------------------------------')
    if(len(cf) != 0):
        print()
        print('*** Corrupted File(s) Found! ***')
        for i in range(len(cf)):
            print(f'{i+1}. {cf[i]}')
        print()
    print(f'Total Samples found : {cnt + test_samples}')
    print(f'Using for Test : {test_samples}')
    print('------------------------------------------------------')
    return dataset_list
     
def test_model(dataset, model_name):
    model = tf.keras.models.load_model(model_name)
    print(model.summary())
    total, cnt = 0. , 0.
    for d in dataset:
        try:
            res = model.evaluate(d)
            total += res[1]
            cnt += 1
        except Exception:
            pass
    print('Avg Accuracy : ', total / cnt)
    return  
    
def train_model_V1(dataset, epochs):  
    model = Sequential([
        LSTM(144, return_sequences = True, input_shape = (None, 12)),
        LSTM(100, return_sequences = False, dropout = 0.25),
        Dense(80, activation = "tanh"),
        Dropout(0.15),
        Dense(36, activation = "softmax")
    ])
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ["accuracy"])
    print(model.summary())
    for ep in range(epochs):
        for i in range(1, len(dataset)):
            model.fit(dataset[i])
        print()
        print(f'Testing after epoch no. {ep}')
        res = model.evaluate(dataset[0])
        model.save('V1model.h5')
        print(f'Current model saved with Acc = {res[1]} (on Test)')
        print()
        
    model.save('V1model.h5')
    
def train_model_V2(dataset, epochs):  
    model = Sequential([
        LSTM(144, return_sequences = False, input_shape = (None, 12), implementation = 1),
        Dense(36, activation = "softmax")
    ])
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ["accuracy"])
    print(model.summary())
    for ep in range(epochs):
        for i in range(1, len(dataset)):
            model.fit(dataset[i])
        print()
        print(f'Testing after epoch no. {ep}')
        model.evaluate(dataset[0])
        model.save('V2model.h5')
        print()
        
    model.save('V2model.h5')
    
if __name__ == "__main__":	
    use_gpu(False)
    tfds_list = input_pipeline(250, 500)
    train_model_V1(tfds_list, 30)
    #test_model(tfds_list, 'model.h5')
