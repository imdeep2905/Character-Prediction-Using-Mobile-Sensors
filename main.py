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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.ticker as ticker
import seaborn as sn
import os

ROOT = Path("./ProcessedDataset3")
y_true = []
y_pred = []
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

CLASSES1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
           '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
           '30', '31', '32', '33', '34', '35']

def use_gpu(gpu):
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def input_pipeline(split, test = False):   
    if test:
        ROOT = Path("./UnseenTest")
    else:
        ROOT = Path("./ProcessedDataset3")
        
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
            
    if(labeled_ds != None):
        dataset_list.append(labeled_ds)
    print('------------------------------------------------------')
    if(len(cf) != 0):
        print()
        print('*** Corrupted File(s) Found! ***')
        for i in range(len(cf)):
            print(f'{i+1}. {cf[i]}')
        print()
    if test:
        print(f'Total Samples found : {cnt}')
        print('Using all samples for testing.')
        print(f'Dataset will have {len(dataset_list)} part(s).')
    else:
        print(f'Total Samples found : {cnt}')
        print('Using all samples for training.')
        print(f'Dataset will have {len(dataset_list)} part(s).')
    print('------------------------------------------------------')
    return dataset_list
     
def test_model(dataset, model_name):
    '''
    #TODO
        Change test_model such that it can do confusion matrix, F1, PR, LOC etc...
    '''
    model = tf.keras.models.load_model(model_name)
    print(model.summary())
    total, cnt = 0. , 0
    '''
    for d in dataset:
        try:
            res = model.evaluate(d)
            total += res[1]
            cnt += 1
        except Exception:
            pass
    '''
    for ds in dataset:
        for X,Y in ds:
            a = np.asarray(Y)
            y_true.append(a.argmax())
            b = model.predict(X)
            y_pred.append(b.argmax())
            cnt+=1
    matrix = confusion_matrix(y_true,y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = sn.heatmap(matrix,cmap='Blues')
    for _,spine in res.spines.items():
        spine.set_visible(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticklabels(['']+CLASSES)
    ax.set_xticklabels(['']+CLASSES)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix")
    plt.show()
    print(classification_report(y_true,y_pred,target_names= CLASSES))
    return  
    
def train_model_V1(dataset, epochs):  
    model = Sequential([
        GRU(144, return_sequences = True, input_shape = (None, 12), recurrent_dropout = 0.2, dropout = 0.3),
        GRU(90, return_sequences = False, recurrent_dropout = 0.1, dropout = 0.25),
        Dense(72, activation = "elu"),
        Dropout(0.4),
        Dense(36, activation = "softmax")
    ])
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ["accuracy"])
    print(model.summary())
    for ep in range(epochs):
        avgacc = 0.
        for ds in dataset:
            result = model.fit(ds)
            with open('log.csv', mode = 'a') as f:
                f.write(f"{ep},{result.history['loss'][0]},{result.history['accuracy'][0]}\n")
                avgacc += result.history['accuracy'][0]
        print()
        print(f'Completed epoch. {ep} ')
        model.save(f'epoch{ep}.h5')
        print(f'Current model saved with Acc = {avgacc / len(dataset)}')
        print()
        
def continue_training(name, dataset, epochs, offset = 0):
    model = keras.models.load_model(name)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ["accuracy"])
    print(model.summary())
    for ep in range(epochs):
        avgacc = 0.
        for ds in dataset:
            result = model.fit(ds)
            with open('log.csv', mode = 'a') as f:
                f.write(f"{ep + offset},{result.history['loss'][0]},{result.history['accuracy'][0]}\n")
                avgacc += result.history['accuracy'][0]
        print()
        print(f'Completed epoch. {ep + offset} ')
        model.save(name)
        print(f'Current model saved with Acc = {avgacc / len(dataset)}')
        print()
    
if __name__ == "__main__":	
    try:
        os.remove('log.csv')
    except Exception:
        pass
    use_gpu(False)
    #DO NOT CHANGE ANYTHING STARTING FROM HERE!
    #continue_training('1.h5', input_pipeline(897), 5, 20)
    #train_model_V1(input_pipeline(897), 50)
    test_model(input_pipeline(576, test = True), 'epoch29.h5')

'''
Record during training:
    # Loss
    # Accuracy
Evaluation methods:
    # Confusion Matrix
    # F1 Score
    # Precision / Recall
    # Acc , Loss
'''