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
from sklearn.neighbors import KNeighborsClassifier

'''
    at testing 5
    IDEA! (1/14)
    - 지금 unsupervised에서는 임베딩값으로만 분류하게끔 하여 cosine distance로 KNN
        - 근데 encoder를 많이 두면 성능이 더 좋아지지 않을까 (더 정보 압축을 하니깐 마지막엔 유용한 정보로 임베딩 구성)
        - encoder를 직렬적으로 많이 두면 당연히 정보 손실이 발생되니깐, skip connection block을 두기
            - 그래서 argument에 --no_skip, --use_skip 둘거임. skip conection을 쓰지 않으면 정보손실로 결과가 안좋게 나오지 않을까 싶음 (아마도)
        - argument에 --num_encoders 두어서 encoder 갯수 사용자에게 받기
    - 실험
        - (1) encoder 수 : 1개(원본), 2개, 3개
        - (2) skip : no_skip, use_skip
        - 그러면 순서쌍은 총 6개 -> 경우의수는 6가지 나옴
'''

class Encoder(Model):
    def __init__(self, in_feat, out_feat, num_encoders=1, use_skip=True):
        super(Encoder, self).__init__()
        self.use_skip = use_skip
        self.num_encoders = num_encoders
        
        for i in range(num_encoders) :
            setattr(self, f'conv1_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
            setattr(self, f'maxpool1_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            setattr(self, f'conv2_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
            setattr(self, f'maxpool2_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            setattr(self, f'conv3_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
        
    def call(self, x, training=False) :
        for i in range(self.num_encoders):
            if self.use_skip and i > 0:
                identity = x
            
            x = getattr(self, f'conv1_{i}')(x)
            x = getattr(self, f'maxpool1_{i}')(x)
            x = getattr(self, f'conv2_{i}')(x)
            x = getattr(self, f'maxpool2_{i}')(x)
            x = getattr(self, f'conv3_{i}')(x)
            
            if self.use_skip and i > 0 : # skip connection block
                x = x + identity
                
        return tf.reduce_mean(x, axis=1)

class CAGE(Model):
    def __init__(self, n_feat, n_cls, proj_dim=0, num_encoders=1, use_skip=True):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        self.enc_A = Encoder(n_feat, 64, num_encoders, use_skip)
        self.enc_G = Encoder(n_feat, 64, num_encoders, use_skip)
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