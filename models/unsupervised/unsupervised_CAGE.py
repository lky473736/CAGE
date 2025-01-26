import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

class DefaultEncoder(Model):
    def __init__(self, in_feat, out_feat, num_encoders=1, use_skip=True):
        super(DefaultEncoder, self).__init__()
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

class TransformerEncoder(Model):
    def __init__(self, in_feat, out_feat, num_heads=8, ff_dim=None, num_blocks=3, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        if ff_dim is None:
            ff_dim = out_feat * 4
            
        self.input_proj = layers.Dense(out_feat)
        self.pos_encoding = layers.Dense(out_feat)
        
        self.num_blocks = num_blocks
        self.head_size = out_feat // num_heads
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        self.output_proj = layers.Dense(out_feat)
    
    def call(self, x, training=False):
        x = self.input_proj(x)
        
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoding = self.pos_encoding(tf.cast(positions, tf.float32))
        x = x + pos_encoding
        
        for _ in range(self.num_blocks):
            x = transformer_encoder(
                x,
                self.head_size,
                self.num_heads,
                self.ff_dim,
                self.dropout
            )
        
        x = tf.reduce_mean(x, axis=1)
        return self.output_proj(x)

class UNetEncoder(Model):
    def __init__(self, in_feat, out_feat):
        super(UNetEncoder, self).__init__()
        
        # encoder path
        self.enc1 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(2)
        self.enc2 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(2)
        self.enc3 = layers.Conv1D(128, 3, activation='relu', padding='same')
        
        # decoder path
        self.up2 = layers.UpSampling1D(2)
        self.dec2 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.up1 = layers.UpSampling1D(2)
        self.dec1 = layers.Conv1D(out_feat, 3, activation='relu', padding='same')
    
    def call(self, x, training=False):
        # encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        
        # decoder + skip connections
        d2 = self.up2(e3)
        d2 = tf.concat([d2, e2], axis=-1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = tf.concat([d1, e1], axis=-1)
        d1 = self.dec1(d1)
        
        return tf.reduce_mean(d1, axis=1)

class CAGE(Model):
    def __init__(self, n_feat, n_cls, proj_dim=0, encoder_type='default', **kwargs):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        
        if encoder_type == 'default' :
            self.enc_A = DefaultEncoder(n_feat, 64, **kwargs)
            self.enc_G = DefaultEncoder(n_feat, 64, **kwargs)
        elif encoder_type == 'transformer' :
            self.enc_A = TransformerEncoder(n_feat, 64, **kwargs)
            self.enc_G = TransformerEncoder(n_feat, 64, **kwargs)
        elif encoder_type == 'unet' :
            self.enc_A = UNetEncoder(n_feat, 64)
            self.enc_G = UNetEncoder(n_feat, 64)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
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


###########################################################################3
# below, ver 1, ver 2

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

# import tensorflow as tf
# from tensorflow.keras import layers, Model
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# class Encoder(Model):
#     def __init__(self, in_feat, out_feat, num_encoders=1, use_skip=True):
#         super(Encoder, self).__init__()
#         self.use_skip = use_skip
#         self.num_encoders = num_encoders
        
#         for i in range(num_encoders) :
#             setattr(self, f'conv1_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
#             setattr(self, f'maxpool1_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
#             setattr(self, f'conv2_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
#             setattr(self, f'maxpool2_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
#             setattr(self, f'conv3_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
        
#     def call(self, x, training=False) :
#         for i in range(self.num_encoders):
#             if self.use_skip and i > 0:
#                 identity = x
            
#             x = getattr(self, f'conv1_{i}')(x)
#             x = getattr(self, f'maxpool1_{i}')(x)
#             x = getattr(self, f'conv2_{i}')(x)
#             x = getattr(self, f'maxpool2_{i}')(x)
#             x = getattr(self, f'conv3_{i}')(x)
            
#             if self.use_skip and i > 0 : # skip connection block
#                 x = x + identity
                
#         return tf.reduce_mean(x, axis=1)

# class CAGE(Model):
#     def __init__(self, n_feat, n_cls, proj_dim=0, num_encoders=1, use_skip=True):
#         super(CAGE, self).__init__()
#         self.proj_dim = proj_dim
#         self.enc_A = Encoder(n_feat, 64, num_encoders, use_skip)
#         self.enc_G = Encoder(n_feat, 64, num_encoders, use_skip)
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