import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class DeepConvLSTM(Model):
    def __init__(self, n_feat, n_cls):
        super(DeepConvLSTM, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        # convs
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
        
        # LSTM
        self.lstm1 = layers.LSTM(128, return_sequences=True)
        self.lstm2 = layers.LSTM(128, return_sequences=True)
        
        self.fc = tf.keras.Sequential([
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
        self.apply_weight_init()
        
    def apply_weight_init(self):
        def orthogonal_init(shape, dtype=None) :
            return tf.keras.initializers.Orthogonal()(shape=shape, dtype=dtype) # orthogonal initial
            
        for layer in self.submodules:
            if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.Dense):
                layer.kernel_initializer = orthogonal_init
                layer.bias_initializer = 'zeros'
            elif isinstance(layer, layers.LSTM):
                for weight in layer.trainable_weights:
                    if 'kernel' in weight.name:
                        weight.assign(orthogonal_init(weight.shape, weight.dtype))
                    elif 'bias' in weight.name:
                        weight.assign(tf.zeros_like(weight))
    
    def call(self, x, training=False):
        # input shape : [batch_size, time_steps, features]
        x = tf.transpose(x, [0, 2, 1])
        
        # convs
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        # LSTMs
        out = self.lstm1(out)
        out = self.lstm2(out)
        
        last = out[:, -1, :]   # getting the last timestep outputs
        out = self.fc(last, training=training) # final classifier
        return out

if __name__ == "__main__":
    model = DeepConvLSTM(n_feat=6, n_cls=6)
    
    sample_input = tf.random.normal([32, 6, 128])  # [batch_size, n_feat, time_steps]
    _ = model(sample_input)
    
    # Count parameters
    total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"Total trainable parameters: {total_params:,}")
    
    '''
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32)'''