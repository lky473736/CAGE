import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class Baseline_CNN(Model):
    def __init__(self, n_feat, n_cls, fc_ch):
        super(Baseline_CNN, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        # Conv1 layer
        self.conv1 = layers.Conv1D(filters=64, kernel_size=5)
        self.conv1_activation = layers.ReLU()
        
        # Conv2 layer
        self.conv2 = layers.Conv1D(filters=64, kernel_size=5)
        self.conv2_activation = layers.ReLU()
        
        # Conv3 layer
        self.conv3 = layers.Conv1D(filters=64, kernel_size=5)
        self.conv3_activation = layers.ReLU()
        
        # Conv4 layer
        self.conv4 = layers.Conv1D(filters=64, kernel_size=5)
        self.conv4_activation = layers.ReLU()
        
        # FC1 layer
        self.fc1 = layers.Dropout(0.5)
        self.fc1_dense = layers.Dense(128)
        self.fc1_activation = layers.ReLU()
        
        # FC2 layer
        self.fc2 = layers.Dropout(0.5)
        self.fc2_dense = layers.Dense(128)
        self.fc2_activation = layers.ReLU()
        
        # FC3 layer
        self.fc3 = layers.Dropout(0.5)
        self.fc3_dense = layers.Dense(n_cls)
        
        # Initialize weights
        self.fc_ch = fc_ch
        self._init_weights()
        
    def _init_weights(self):
        # dummy input
        dummy_input = tf.random.normal((1, self.fc_ch, self.n_feat))
        self.call(dummy_input)
        
        # orthogonal initialization
        for layer in self.layers:
            if isinstance(layer, (layers.Conv1D, layers.Dense)):
                orthogonal_initializer = tf.keras.initializers.Orthogonal()
                # https://076923.github.io/posts/AI-8/
                # https://arxiv.org/abs/2406.01755
                # 직교 초기화를 하는게 맞나도 의문이긴 한데...
                weights = orthogonal_initializer(layer.weights[0].shape)
                layer.set_weights([
                    weights,
                    tf.zeros(layer.weights[1].shape)
                ])

    def call(self, x, training=False):
        # tf expects shape (batch_size, time_steps, channels)
        # but input is (batch_size, channels, time_steps)
        x = tf.transpose(x, perm=[0, 2, 1]) # 여기서 transpose를 하긴 했는데 이렇게 해서 channel size unmatching오류가 났을 수도 있겠음
        
        # conv
        x = self.conv1(x)
        x = self.conv1_activation(x)
        
        x = self.conv2(x)
        x = self.conv2_activation(x)
        
        x = self.conv3(x)
        x = self.conv3_activation(x)
        
        x = self.conv4(x)
        x = self.conv4_activation(x)
        
        # flatten
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        
        # fcc
        x = self.fc1(x)
        x = self.fc1_dense(x)
        x = self.fc1_activation(x)
        
        x = self.fc2(x)
        x = self.fc2_dense(x)
        x = self.fc2_activation(x)
        
        x = self.fc3(x)
        x = self.fc3_dense(x)
        
        return x

if __name__ == "__main__":
    model = Baseline_CNN(6, 6, 128)
    
    # dummy input
    dummy_input = tf.random.normal((1, 128, 6))
    _ = model(dummy_input)
    
    # counting params
    n_all = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
