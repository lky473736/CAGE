import tensorflow as tf
from tensorflow.keras import Model, layers

# Implementation of "LSTM-CNN Architecture for Human Activity Recognition" by Kun Xia et al.
class LSTMConvNet(Model):
    def __init__(self, n_feat, n_cls):
        super(LSTMConvNet, self).__init__()
        self.n_feat = n_feat
        
        # LSTM layers
        self.lstm1 = layers.LSTM(units=32, return_sequences=True)
        self.lstm2 = layers.LSTM(units=32, return_sequences=True)
        
        # Convolutional layers
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
        
        # Final fully connected layer with dropout
        self.dropout = layers.Dropout(0.5)
        self.fc = layers.Dense(n_cls)
    
    def call(self, x, training=False):
        # Input shape: [batch_size, n_features, window_size]
        
        # Reshape for LSTM [batch, time, features]
        x = tf.transpose(x, perm=[0, 2, 1])
        
        # LSTM layers
        out = self.lstm1(x)
        out = self.lstm2(out)
        
        # Reshape for Conv1D [batch, time, features]
        out = tf.transpose(out, perm=[0, 2, 1])
        
        # Conv layers
        out = self.conv1(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        
        # Global average pooling
        out = tf.reduce_mean(out, axis=-1, keepdims=True)
        
        # Batch normalization
        out = self.bn(out, training=training)
        
        # Flatten
        out = tf.reshape(out, [tf.shape(out)[0], -1])
        
        # Final dense layer with dropout
        out = self.dropout(out, training=training)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    model = LSTMConvNet(6, 6)
    # Build model with sample input
    sample_input = tf.random.normal((1, 6, 128))  # [batch_size, n_features, window_size]
    _ = model(sample_input)
    print("Total trainable parameters:", model.count_params())