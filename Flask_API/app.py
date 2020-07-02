import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import flask
import io

app = flask.Flask(__name__)
model = None

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

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

def load_model():
    global model 
    model = tf.keras.models.load_model('2D.h5')

def prepare_csv(csv):
    X = (pd.read_csv(io.BytesIO(csv)))
    X = X.filter(take, axis=1)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    X = np.expand_dims(X, axis = 0)
    return X

@app.route("/predict", methods = ["POST"])
def predict():
    data = {"success" : False}
    try:
        if flask.request.method == "POST":
            if flask.request.files.get("csv"):
                csv = flask.request.files["csv"].read()
                csv = prepare_csv(csv)
                preds = np.asfarray(model.predict(csv))
                preds = preds[0]
                preds = preds.argsort()[-5:][::-1]
                data["predictions"] = []
                for p in preds:
                    data["predictions"].append(CLASSES[p])
                data["success"] = True
    except Exception:
        return flask.jsonify(data)
    
    return flask.jsonify(data)
    
if __name__ == "__main__":
    load_model()
    app.run(debug = True)