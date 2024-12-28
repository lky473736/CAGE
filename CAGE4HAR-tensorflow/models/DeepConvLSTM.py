import tensorflow as tf
from tensorflow.keras import Model, layers, initializers

# Implementation of "Deep ConvLSTM with self-attention for human activity decoding using wearables"
class DeepConvLSTM(Model):
    def __init__(self, n_feat, n_cls):
        super(DeepConvLSTM, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        
        # Orthogonal initializer to match PyTorch's orthogonal initialization
        self.orthogonal = initializers.Orthogonal()
        self.zeros = initializers.Zeros()
        
        # Convolutional layers
        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, kernel_initializer=self.orthogonal,
                         bias_initializer=self.zeros),
            layers.ReLU()
        ])
        
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, kernel_initializer=self.orthogonal,
                         bias_initializer=self.zeros),
            layers.ReLU()
        ])
        
        self.conv3 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, kernel_initializer=self.orthogonal,
                         bias_initializer=self.zeros),
            layers.ReLU()
        ])
        
        self.conv4 = tf.keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=5, kernel_initializer=self.orthogonal,
                         bias_initializer=self.zeros),
            layers.ReLU()
        ])
        
        # LSTM layers
        self.lstm1 = layers.LSTM(units=128, return_sequences=True,
                                kernel_initializer=self.orthogonal,
                                recurrent_initializer=self.orthogonal,
                                bias_initializer=self.zeros)
        
        self.lstm2 = layers.LSTM(units=128, return_sequences=True,
                                kernel_initializer=self.orthogonal,
                                recurrent_initializer=self.orthogonal,
                                bias_initializer=self.zeros)
        
        # Final layers
        self.dropout = layers.Dropout(0.5)
        self.fc = layers.Dense(n_cls, kernel_initializer=self.orthogonal,
                             bias_initializer=self.zeros)

    def call(self, x, training=False):
        # Conv layers
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        # Reshape for LSTM (TensorFlow uses [batch, time, features])
        out = tf.transpose(out, perm=[0, 2, 1])
        
        # LSTM layers
        out = self.lstm1(out)
        out = self.lstm2(out)
        
        # Get last timestep
        last = out[:, -1, :]
        
        # Final dense layer with dropout
        out = self.dropout(last, training=training)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    model = DeepConvLSTM(6, 6)
    # Build model with sample input
    sample_input = tf.random.normal((1, 6, 128))  # [batch_size, n_features, window_size]
    _ = model(sample_input)
    print("Total trainable parameters:", model.count_params())