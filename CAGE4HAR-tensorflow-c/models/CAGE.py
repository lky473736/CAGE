import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class Encoder(Model):
    def __init__(self, in_feat, out_feat):
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
            layers.ReLU(),
        ])
    
    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return tf.reduce_mean(x, axis=1)  
        # mean pooling

class CAGE(Model):
    def __init__(self, n_feat, n_cls, proj_dim=0):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        
        # encoder
        self.enc_A = Encoder(n_feat, 64)
        self.enc_G = Encoder(n_feat, 64)
        
        if self.proj_dim > 0:
            self.proj_A = layers.Dense(proj_dim, use_bias=False)
            self.proj_G = layers.Dense(proj_dim, use_bias=False)
        
        self.temperature = tf.Variable(initial_value=0.07, trainable=True)
        
        # classifier
        self.classifier = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
    def call(self, inputs, training=False, return_feat=False):
        x_accel, x_gyro = inputs
        
        # (batch_size, channels, time_steps) -> (batch_size, time_steps, channels)
        x_accel = tf.transpose(x_accel, perm=[0, 2, 1])
        x_gyro = tf.transpose(x_gyro, perm=[0, 2, 1])
        
        # get feature from encoder (feature extraction)
        f_accel = self.enc_A(x_accel, training=training)
        f_gyro = self.enc_G(x_gyro, training=training)
        
        # 가속도계 + 자이로스코프 (concatenation) => classifier
        concat_features = tf.concat([f_accel, f_gyro], axis=-1)
        out = self.classifier(concat_features, training=training)
        
        # project
        if self.proj_dim > 0:
            e_accel = self.proj_A(f_accel)
            e_gyro = self.proj_G(f_gyro)
        else:
            e_accel = f_accel
            e_gyro = f_gyro
        
        # embedding space를 normalize
        e_accel_norm = tf.nn.l2_normalize(e_accel, axis=1)
        e_gyro_norm = tf.nn.l2_normalize(e_gyro, axis=1)
        
        '''
            logits가 여기서 왜 필요한 지는 의문, 하지만 pytorch에서 사용해서 일단은 tf 형식으로 고쳐놓은 상태
        '''
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

class CAGE_EarlyFusion(Model) : # early fusion model
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
        
    def call(self, x, training=False, return_feat=False):
        x = tf.transpose(x, perm=[0, 2, 1]) # transpose (tf의 특성상)
        out = self.encoder(x, training=training)
        out = self.classifier(out, training=training)
        return out

if __name__ == "__main__":
    model = CAGE_EarlyFusion(3, 6)
    dummy_input = tf.random.normal((1, 3, 128))
    _ = model(dummy_input)
    n_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print(f"CAGE_EarlyFusion trainable parameters: {n_params}")
    
    model = CAGE(3, 6, 64)
    dummy_accel = tf.random.normal((1, 3, 128))
    dummy_gyro = tf.random.normal((1, 3, 128))
    _ = model((dummy_accel, dummy_gyro))
    n_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print(f"CAGE trainable parameters: {n_params}")