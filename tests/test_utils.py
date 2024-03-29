from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import numpy as np

def simple_model():
    model = Sequential()
    model.add(Dense(2, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def rmse(y_true, y_pred): 
    y_true = K.cast(y_true, 'float32')
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class SimpleGenerator(Sequence):
    def __init__(self, x, y, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x, y