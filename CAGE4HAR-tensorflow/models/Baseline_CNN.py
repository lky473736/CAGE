import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class Baseline_CNN(Model):
    def __init__(self, n_feat, n_cls, fc_ch):
        super(Baseline_CNN, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        # Conv1 layer
        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, input_shape=(None, n_feat)),
            layers.ReLU()
        ])
        
        # Conv2 layer
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5),
            layers.ReLU()
        ])
        
        # Conv3 layer
        self.conv3 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5),
            layers.ReLU()
        ])
        
        # Conv4 layer
        self.conv4 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5),
            layers.ReLU()
        ])
        
        # FC1 layer
        self.fc1 = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.ReLU()
        ])
        
        # FC2 layer
        self.fc2 = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.ReLU()
        ])
        
        # FC3 layer
        self.fc3 = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
        # Initialize weights
        self.fc_ch = fc_ch
        self._init_weights()
        
    def _init_weights(self):
        # Create dummy input to build the model
        dummy_input = tf.random.normal((1, self.fc_ch, self.n_feat))
        self.call(dummy_input)
        
        # Apply orthogonal initialization
        for layer in self.layers:
            if isinstance(layer, tf.keras.Sequential):
                for sublayer in layer.layers:
                    if isinstance(sublayer, (layers.Conv1D, layers.Dense)):
                        # Orthogonal initialization
                        orthogonal_initializer = tf.keras.initializers.Orthogonal()
                        weights = orthogonal_initializer(sublayer.weights[0].shape)
                        sublayer.set_weights([
                            weights,
                            tf.zeros(sublayer.weights[1].shape)
                        ])

    def call(self, x, training=False):
        # TensorFlow expects shape (batch_size, time_steps, channels)
        # but input is (batch_size, channels, time_steps)
        x = tf.transpose(x, perm=[0, 2, 1])
        
        # Convolutional layers
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        # Flatten
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        
        # Fully connected layers
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        
        return x

if __name__ == "__main__":
    model = Baseline_CNN(6, 6, 128)
    
    # Build model with dummy input
    dummy_input = tf.random.normal((1, 128, 6))
    _ = model(dummy_input)
    
    # Count parameters
    n_all = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))