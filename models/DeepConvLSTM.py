import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
import numpy as np

class DeepConvLSTM(Model):
    def __init__(self, n_feat, n_cls):
        super(DeepConvLSTM, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        kernel_init = initializers.Orthogonal()
        
        self.conv1 = layers.Conv1D(filters=64, kernel_size=5, 
                                 activation='relu',
                                 kernel_initializer=kernel_init,
                                 bias_initializer='zeros')
        self.conv2 = layers.Conv1D(filters=64, kernel_size=5,
                                 activation='relu',
                                 kernel_initializer=kernel_init,
                                 bias_initializer='zeros')
        self.conv3 = layers.Conv1D(filters=64, kernel_size=5,
                                 activation='relu',
                                 kernel_initializer=kernel_init,
                                 bias_initializer='zeros')
        self.conv4 = layers.Conv1D(filters=64, kernel_size=5,
                                 activation='relu',
                                 kernel_initializer=kernel_init,
                                 bias_initializer='zeros')
        
        self.lstm1 = layers.LSTM(128, return_sequences=True,
                               kernel_initializer=kernel_init,
                               recurrent_initializer=kernel_init,
                               bias_initializer='zeros')
        self.lstm2 = layers.LSTM(128, return_sequences=True,
                               kernel_initializer=kernel_init,
                               recurrent_initializer=kernel_init,
                               bias_initializer='zeros')
        
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(n_cls,
                                kernel_initializer=kernel_init,
                                bias_initializer='zeros')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = tf.transpose(x, [0, 2, 1]) 
        x = self.lstm1(x)
        x = self.lstm2(x)
        
        x = x[:, -1, :]
        
        x = self.dropout(x, training=training)
        x = self.dense(x)
        
        return x

if __name__ == "__main__":
    model = DeepConvLSTM(6, 6)
    print(sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables))