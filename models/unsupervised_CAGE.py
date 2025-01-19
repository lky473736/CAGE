# import tensorflow as tf
# from tensorflow.keras import layers, Model
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# class Encoder(Model):
#     def __init__(self, in_feat, out_feat):
#         super(Encoder, self).__init__()
        
#         self.conv1 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
#         self.maxpool1 = layers.MaxPooling1D(pool_size=2, padding='same')
#         self.conv2 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
#         self.maxpool2 = layers.MaxPooling1D(pool_size=2, padding='same')
#         self.conv3 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
        
#     def call(self, x, training=False):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.maxpool2(x)
#         x = self.conv3(x)
#         return tf.reduce_mean(x, axis=1)

# class CAGE(Model):
#     def __init__(self, n_feat, n_cls, proj_dim=0):
#         super(CAGE, self).__init__()
#         self.proj_dim = proj_dim
#         self.enc_A = Encoder(n_feat, 64)
#         self.enc_G = Encoder(n_feat, 64)
#         if self.proj_dim > 0:
#             self.proj_A = layers.Dense(proj_dim, use_bias=False)
#             self.proj_G = layers.Dense(proj_dim, use_bias=False)
#         self.temperature = tf.Variable(0.07, trainable=True)
        
#     def call(self, x_accel, x_gyro, return_feat=False, training=False):
#         f_accel = self.enc_A(x_accel, training=training)
#         f_gyro = self.enc_G(x_gyro, training=training)
        
#         if self.proj_dim > 0:
#             e_accel = self.proj_A(f_accel)
#             e_gyro = self.proj_G(f_gyro)
#         else:
#             e_accel = f_accel
#             e_gyro = f_gyro
            
#         e_accel = tf.math.l2_normalize(e_accel, axis=1)
#         e_gyro = tf.math.l2_normalize(e_gyro, axis=1)
        
#         logits = tf.matmul(e_accel, e_gyro, transpose_b=True) * tf.exp(self.temperature)
        
#         if return_feat:
#             return logits, (f_accel, f_gyro)
        
#         return logits

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Encoder(Model):
    def __init__(self, in_feat, out_feat, num_encoders=1, use_skip=True):
        super(Encoder, self).__init__()
        self.use_skip = use_skip
        self.num_encoders = num_encoders
        
        for i in range(num_encoders):
            setattr(self, f'conv1_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
            setattr(self, f'maxpool1_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            setattr(self, f'conv2_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
            setattr(self, f'maxpool2_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            setattr(self, f'conv3_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
        
    def call(self, x, training=False):
        for i in range(self.num_encoders):
            if self.use_skip and i > 0:
                identity = x
            
            x = getattr(self, f'conv1_{i}')(x)
            x = getattr(self, f'maxpool1_{i}')(x)
            x = getattr(self, f'conv2_{i}')(x)
            x = getattr(self, f'maxpool2_{i}')(x)
            x = getattr(self, f'conv3_{i}')(x)
            
            if self.use_skip and i > 0:
                x = x + identity
                
        return tf.reduce_mean(x, axis=1)

def nt_xent_loss(z_i, z_j, temperature):
    batch_size = tf.shape(z_i)[0]
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    
    sim_matrix = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    pos_sim = tf.linalg.diag_part(sim_matrix)
    
    loss = -tf.reduce_mean(
        pos_sim - tf.reduce_logsumexp(sim_matrix, axis=1))
    
    return loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(0.0, margin + pos_dist - neg_dist)
    return tf.reduce_mean(loss)

def get_hard_negatives(anchor_features, all_features, k=10):
    distances = tf.reduce_sum(tf.square(
        anchor_features[:, tf.newaxis] - all_features), axis=2)
    _, hard_negative_indices = tf.nn.top_k(-distances, k=k)
    return hard_negative_indices

class CAGE(Model):
    def __init__(self, n_feat, proj_dim=0, num_encoders=1, use_skip=True, loss_type='nt_xent'):
        super(CAGE, self).__init__()
        self.loss_type = loss_type
        self.proj_dim = proj_dim
        self.enc_A = Encoder(n_feat, 64, num_encoders, use_skip)
        self.enc_G = Encoder(n_feat, 64, num_encoders, use_skip)
        
        if self.proj_dim > 0:
            self.proj_A = layers.Dense(proj_dim, use_bias=False)
            self.proj_G = layers.Dense(proj_dim, use_bias=False)
            
        self.temperature = tf.Variable(0.07, trainable=True,
                                     constraint=lambda x: tf.clip_by_value(x, 0.01, 0.5))
        
    def encode(self, x_accel, x_gyro, training=False):
        f_accel = self.enc_A(x_accel, training=training)
        f_gyro = self.enc_G(x_gyro, training=training)
        
        if self.proj_dim > 0:
            f_accel = self.proj_A(f_accel)
            f_gyro = self.proj_G(f_gyro)
            
        return f_accel, f_gyro
    
    def call(self, x_accel, x_gyro, return_feat=False, training=False):
        f_accel, f_gyro = self.encode(x_accel, x_gyro, training)
        
        e_accel = tf.math.l2_normalize(f_accel, axis=1)
        e_gyro = tf.math.l2_normalize(f_gyro, axis=1)
        logits = tf.matmul(e_accel, e_gyro, transpose_b=True) / self.temperature
        
        if return_feat:
            if self.loss_type == 'nt_xent':
                return logits, (e_accel, e_gyro)
            elif self.loss_type == 'triplet':
                return logits, (f_accel, f_gyro)
            else:  # default
                return logits, (e_accel, e_gyro)
        
        return logits