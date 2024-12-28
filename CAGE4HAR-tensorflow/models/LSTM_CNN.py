# Implementation of "LSTM-CNN Architecture for Human Activity Recognition" by Kun Xia et al. (IEEE Access, 2020)

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class LSTMConvNet(Model):
    def __init__(self, n_feat, n_cls):
        super(LSTMConvNet, self).__init__()
        self.n_feat = n_feat
        
        # LSTMs
        self.lstm1 = layers.LSTM(32, return_sequences=True)
        self.lstm2 = layers.LSTM(32, return_sequences=True)
        
        # convs
        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, strides=2),
            layers.ReLU()
        ])
        
        self.maxpool = layers.MaxPool1D(pool_size=2, strides=2)
        
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(filters=128, kernel_size=3, strides=1),
            layers.ReLU()
        ])
        
        self.bn = layers.BatchNormalization()
        self.fc = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
    def call(self, x, training=False):
        # [batch_size, features, time_steps] -> [batch_size, time_steps, features]
        x = tf.transpose(x, [0, 2, 1])
        
        # LSTM layers
        out = self.lstm1(x)
        out = self.lstm2(out)
        
        # [batch_size, time_steps, features] -> [batch_size, features, time_steps]
        out = tf.transpose(out, [0, 2, 1])
        
        # convs
        out = self.conv1(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        # global average pooling
        out = tf.reduce_mean(out, axis=-1, keepdims=True)
        # batch normalization
        out = self.bn(out, training=training)
        
        # flatten 
        out = tf.reshape(out, [tf.shape(out)[0], -1]) # flatten
        out = self.fc(out, training=training) # dense
        
        return out

if __name__ == "__main__" :
    model = LSTMConvNet(n_feat=6, n_cls=6)
    
    sample_input = tf.random.normal([32, 6, 128])  # [batch_size, n_feat, time_steps]
    _ = model(sample_input)
    
    total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"Total trainable parameters: {total_params:,}")
    