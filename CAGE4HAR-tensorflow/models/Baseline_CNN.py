import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class BaselineCNN(Model) :
    def __init__(self, n_feat, n_cls, fc_ch) :
        super(BaselineCNN, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, input_shape=(None, n_feat)),
            layers.ReLU()
        ])
        
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5),
            layers.ReLU()
        ])
        
        self.conv3 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5),
            layers.ReLU()
        ])
        
        self.conv4 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5),
            layers.ReLU()
        ])
        
        self.fc1 = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.ReLU()
        ])
        
        self.fc2 = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.ReLU()
        ])
        
        self.fc3 = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
        self.apply_weight_init()
    
    def call(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten
        
        # fc == dense
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    
    def apply_weight_init(self) :
        def orthogonal_init(shape, dtype=None) : 
            return tf.keras.initializers.Orthogonal()(shape=shape, dtype=dtype)
        
        for layer in self.layers :
            if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.Dense) :
                layer.kernel_initializer = orthogonal_init
                layer.bias_initializer = 'zeros'

if __name__ == "__main__" :
    model = BaselineCNN(n_feat=6, n_cls=6, fc_ch=128)
    sample_input = tf.random.normal([1, 128, 6]) # sample model of input
    _ = model(sample_input)
    
    total_params = sum([np.prod(var.shape) for var in model.trainable_variables]) # number of param
    print(f"Parameter Count: all {total_params:,d}; trainable {total_params:,d}")