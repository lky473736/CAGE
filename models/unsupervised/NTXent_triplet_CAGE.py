import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import scipy as sci

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

'''
    nt_xent_loss: 같은 동작에서 gyro랑 accer의 positive pair을 사용함
    triplet_loss: 
    - Anchor: 현재 샘플
    - Positive: 같은 class의 다른 시점 데이터
    - Negative: 다른 class의 데이터
    - Margin: posi와 nega의 거리 조절
'''
# class CAGE(Model):
#     def __init__(self, n_feat, proj_dim=0, loss_type='nt_xent'):
#         super(CAGE, self).__init__()
#         self.loss_type = loss_type
#         self.proj_dim = proj_dim
#         self.enc_A = Encoder(n_feat, 64)
#         self.enc_G = Encoder(n_feat, 64)
        
#         if self.proj_dim > 0:
#             self.proj_A = layers.Dense(proj_dim, use_bias=False)
#             self.proj_G = layers.Dense(proj_dim, use_bias=False)
            
#         self.temperature = tf.Variable(0.07, trainable=True)
        
#     def encode(self, x_accel, x_gyro, training=False):
#         f_accel = self.enc_A(x_accel, training=training)
#         f_gyro = self.enc_G(x_gyro, training=training)
        
#         if self.proj_dim > 0:
#             f_accel = self.proj_A(f_accel)
#             f_gyro = self.proj_G(f_gyro)
            
#         return f_accel, f_gyro
    
#     def nt_xent_loss(self, f_accel, f_gyro, temperature=0.07):
#         batch_size = tf.shape(f_accel)[0]
#         z_i = tf.math.l2_normalize(f_accel, axis=1)
#         z_j = tf.math.l2_normalize(f_gyro, axis=1)
        
#         logits = tf.matmul(z_i, z_j, transpose_b=True) / temperature
#         labels = tf.range(batch_size)
        
#         loss = tf.keras.losses.sparse_categorical_crossentropy(
#             labels, logits, from_logits=True)
#         return tf.reduce_mean(loss)
    
#     def triplet_loss(self, anchor, positive, negative, margin=1.0):
#         pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
#         neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
#         loss = tf.maximum(0.0, margin + pos_dist - neg_dist)
#         return tf.reduce_mean(loss)
    
#     def call(self, x_accel, x_gyro, return_feat=False, training=False):
#         f_accel, f_gyro = self.encode(x_accel, x_gyro, training)
        
#         if self.loss_type == 'nt_xent':
#             e_accel = tf.math.l2_normalize(f_accel, axis=1)
#             e_gyro = tf.math.l2_normalize(f_gyro, axis=1)
#             logits = tf.matmul(e_accel, e_gyro, transpose_b=True) / self.temperature
            
#             if return_feat:
#                 return logits, (f_accel, f_gyro)
#             return logits
        
#         if return_feat:
#             return None, (f_accel, f_gyro)
#         return f_accel, f_gyro

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

# def nt_xent_loss(z_i, z_j, temperature=0.07):
#    batch_size = tf.shape(z_i)[0]
#    z_i = tf.math.l2_normalize(z_i, axis=1)
#    z_j = tf.math.l2_normalize(z_j, axis=1)
#    logits = tf.matmul(z_i, z_j, transpose_b=True) / temperature
#    labels = tf.range(batch_size)
#    loss = tf.keras.losses.sparse_categorical_crossentropy(labels,
#        logits, from_logits=True)
#    return tf.reduce_mean(loss)

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

class CAGE(Model):
   def __init__(self, n_feat, proj_dim=0, loss_type='nt_xent'):
       super(CAGE, self).__init__()
       self.loss_type = loss_type
       self.proj_dim = proj_dim
       self.enc_A = Encoder(n_feat, 64)
       self.enc_G = Encoder(n_feat, 64)
       
       if self.proj_dim > 0:
           self.proj_A = layers.Dense(proj_dim, use_bias=False)
           self.proj_G = layers.Dense(proj_dim, use_bias=False)
           
    #    self.temperature = tf.Variable(0.07, trainable=True)
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
       
       if self.loss_type == 'nt_xent':
           e_accel = tf.math.l2_normalize(f_accel, axis=1)
           e_gyro = tf.math.l2_normalize(f_gyro, axis=1)
           logits = tf.matmul(e_accel, e_gyro, transpose_b=True) / self.temperature
           
           if return_feat:
               return logits, (e_accel, e_gyro)
           return logits
       
       if return_feat:
           return None, (f_accel, f_gyro)
       return f_accel, f_gyro
   
   def compute_loss(self, x_accel, x_gyro):
       if self.loss_type == 'nt_xent':
           f_accel, f_gyro = self.encode(x_accel, x_gyro)
           return nt_xent_loss(f_accel, f_gyro, self.temperature)
       
       else:  # triplet loss
           f_anchor, _ = self.call(x_accel, x_accel)
           f_positive, _ = self.call(x_accel, x_gyro) 
           batch_size = tf.shape(x_accel)[0]
           neg_idx = tf.random.shuffle(tf.range(batch_size))
           f_negative, _ = self.call(tf.gather(x_accel, neg_idx), 
                                   tf.gather(x_gyro, neg_idx)) 
           return triplet_loss(f_anchor, f_positive, f_negative)