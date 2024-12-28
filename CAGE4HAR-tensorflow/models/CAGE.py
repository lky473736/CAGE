import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class Encoder(Model) :
    def __init__(self, in_feat, out_feat) :
        super(Encoder, self).__init__()
        
        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(filters=out_feat, kernel_size=3, strides=1, padding='same'),
            layers.ReLU(),
            layers.MaxPool1D(pool_size=2)
        ])
        
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(filters=out_feat, kernel_size=3, strides=1, padding='same'),
            layers.ReLU(),
            layers.MaxPool1D(pool_size=2)
        ])
        
        self.conv3 = tf.keras.Sequential([
            layers.Conv1D(filters=out_feat, kernel_size=3, strides=1, padding='same'),
            layers.ReLU()
        ])
        
    def call(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return tf.reduce_mean(x, axis=-2)  

class CAGE(Model) :
    def __init__(self, n_feat, n_cls, proj_dim=0) :
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        
        self.enc_A = Encoder(n_feat, 64)
        self.enc_G = Encoder(n_feat, 64)
        
        if self.proj_dim > 0 :
            self.proj_A = layers.Dense(proj_dim, use_bias=False)
            self.proj_G = layers.Dense(proj_dim, use_bias=False)
        
        self.temperature = tf.Variable(initial_value=[0.07], trainable=True)
        
        self.classifier = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
    def call(self, inputs, return_feat=False, training=None) :
        x_accel, x_gyro = inputs
        
        f_accel = self.enc_A(x_accel)
        f_gyro = self.enc_G(x_gyro)
        
        out = self.classifier(tf.concat([f_accel, f_gyro], axis=-1), training=training)
        
        if self.proj_dim > 0 :
            e_accel = self.proj_A(f_accel)
            e_gyro = self.proj_G(f_gyro)
        else :
            e_accel = f_accel
            e_gyro = f_gyro
        
        # normalization 
        e_accel_norm = tf.nn.l2_normalize(e_accel, axis=-1)
        e_gyro_norm = tf.nn.l2_normalize(e_gyro, axis=-1)
        logits = tf.matmul(e_accel_norm, e_gyro_norm, transpose_b=True) * tf.exp(self.temperature)
        
        if return_feat :
            return logits, out, (e_accel_norm, e_gyro_norm)
        return logits, out
    
    def freeze_enc(self) :
        self.enc_A.trainable = False
        self.enc_G.trainable = False
        if self.proj_dim > 0:
            self.proj_A.trainable = False
            self.proj_G.trainable = False

class CAGE_EarlyFusion(Model) :
    def __init__(self, n_feat, n_cls):
        super(CAGE_EarlyFusion, self).__init__()
        self.encoder = Encoder(n_feat, 128)
        self.classifier = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
    
    def call(self, x, return_feat=False, training=None):
        out = self.encoder(x)
        out = self.classifier(out, training=training)
        return out

if __name__ == "__main__" :
    # CAGE (Early Fusion)
    model = CAGE_EarlyFusion(3, 6)
    sample_input = tf.random.normal([1, 128, 3])
    _ = model(sample_input)
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"CAGE_EarlyFusion trainable parameters: {trainable_params}")
    
    # CAGE
    model = CAGE(3, 6, 64)
    sample_accel = tf.random.normal([1, 128, 3])
    sample_gyro = tf.random.normal([1, 128, 3])
    _ = model((sample_accel, sample_gyro))
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"CAGE trainable parameters: {trainable_params}")