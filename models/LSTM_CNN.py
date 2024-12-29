import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class LSTMConvNet(Model) :
    def __init__(self, n_feat, n_cls) :
        super(LSTMConvNet, self).__init__()
        self.n_feat = n_feat
        
        self.lstm1 = layers.LSTM(32, return_sequences=True)
        self.lstm2 = layers.LSTM(32, return_sequences=True)
        self.conv1 = layers.Conv1D(filters=64, kernel_size=5, 
                                 strides=2, activation='relu')
        self.maxpool = layers.MaxPooling1D(pool_size=2, strides=2)
        self.conv2 = layers.Conv1D(filters=128, kernel_size=3,
                                 strides=1, activation='relu')
        self.bn = layers.BatchNormalization()
        
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(n_cls)

    def call(self, x, training=False):
        x = tf.transpose(x, [0, 2, 1])
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = tf.transpose(x, [0, 2, 1])
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = tf.reduce_mean(x, axis=1, keepdims=True) # global average pooling
        
        # bn
        x = self.bn(x, training=training)
        x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten
        x = self.dropout(x, training=training)
        x = self.dense(x) # last layer
        
        return x

if __name__ == "__main__":
    model = LSTMConvNet(6, 6)
    print(sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables))