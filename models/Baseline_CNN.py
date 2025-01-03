import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Baseline_CNN(Model):
    def __init__(self, n_feat, n_cls, fc_ch):
        super(Baseline_CNN, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        '''
            conv1 -> conv2 -> conv3 -> conv4 -> dr1 -> dense -> dr2 -> dense -> dr3 -> dense
        '''

        self.conv1 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same') # add padding = 'same'
        ''''    
            difference by tensorflow and pytorch
            -> padding is added at tensorflow because of size unmatching probs
        '''
        self.conv2 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')
        self.conv3 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')
        self.conv4 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')
        
        self.dropout1 = layers.Dropout(0.5) # why 0.5 == learning tightly
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(n_cls)
        
        self._init_weights() # initialization (orthogonal)
        
    def _init_weights(self):
        for layer in self.layers : # for all layer
            if isinstance(layer, (layers.Conv1D, layers.Dense)) : 
                layer.kernel_initializer = tf.keras.initializers.Orthogonal() 
                # apply orthogonal -> i dont know why this needs
                layer.bias_initializer = 'zeros'

    def call(self, x, training=False) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) 
        x = tf.reshape(x, [tf.shape(x)[0], -1]) # must reshape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # dense layers and dropoout
        x = self.dropout1(x, training=training)
        x = self.fc1(x)
        x = self.dropout2(x, training=training)
        x = self.fc2(x)
        x = self.dropout3(x, training=training)
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    model = Baseline_CNN(6, 6, 128)
    n_all = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print("Parameter Count: all {:,d}".format(n_all))