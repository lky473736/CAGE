import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class Encoder(Model):
    def __init__(self, in_feat, out_feat):
        super(Encoder, self).__init__()
        
        self.conv1 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
        self.maxpool1 = layers.MaxPooling1D(pool_size=2, padding='same')
        self.conv2 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
        self.maxpool2 = layers.MaxPooling1D(pool_size=2, padding='same')
        self.conv3 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        return tf.reduce_mean(x, axis=1)

class CAGE(Model):
    def __init__(self, n_feat, n_cls, proj_dim=0):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        self.enc_A = Encoder(n_feat, 64)
        self.enc_G = Encoder(n_feat, 64)
        if self.proj_dim > 0:
            self.proj_A = layers.Dense(proj_dim, use_bias=False)
            self.proj_G = layers.Dense(proj_dim, use_bias=False)
        self.temperature = tf.Variable(0.07, trainable=True)
        
    def call(self, x_accel, x_gyro, return_feat=False, training=False):
        f_accel = self.enc_A(x_accel, training=training)
        f_gyro = self.enc_G(x_gyro, training=training)
        
        if self.proj_dim > 0:
            e_accel = self.proj_A(f_accel)
            e_gyro = self.proj_G(f_gyro)
        else:
            e_accel = f_accel
            e_gyro = f_gyro
            
        e_accel = tf.math.l2_normalize(e_accel, axis=1)
        e_gyro = tf.math.l2_normalize(e_gyro, axis=1)
        
        logits = tf.matmul(e_accel, e_gyro, transpose_b=True) * tf.exp(self.temperature)
        
        if return_feat:
            return logits, (f_accel, f_gyro)
        
        return logits
