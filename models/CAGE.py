import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Encoder(Model):
    def __init__(self, in_feat, out_feat):
        super(Encoder, self).__init__()
        
        '''
            key point here is that in tf, activation functions can be passed as an argument to a layer, whereas in pytorch, 
            even a single activation like "relu" is treated as a separate layer. this was quite confusing at first
        '''
        self.conv1 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu') 
        self.maxpool1 = layers.MaxPooling1D(pool_size=2, padding='same')
        self.conv2 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
        self.maxpool2 = layers.MaxPooling1D(pool_size=2, padding='same')
        self.conv3 = layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu')
        
    def call(self, x, training=False) :
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        return tf.reduce_mean(x, axis=1) # pooling 
        # https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221164644393
        # https://steemit.com/kr-steemit/@beseeyong/tensor-part-3

class CAGE(Model) :
    def __init__(self, n_feat, n_cls, proj_dim=0):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        self.enc_A = Encoder(n_feat, 64)
        self.enc_G = Encoder(n_feat, 64)
        if self.proj_dim > 0:
            self.proj_A = layers.Dense(proj_dim, use_bias=False)
            self.proj_G = layers.Dense(proj_dim, use_bias=False)
        self.temperature = tf.Variable(0.07, trainable=True)
        
        self.classifier = tf.keras.Sequential([ # classifier (is at last in this model)
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
    def call(self, x_accel, x_gyro, return_feat=False, training=False):
        f_accel = self.enc_A(x_accel, training=training)
        f_gyro = self.enc_G(x_gyro, training=training)
        if self.proj_dim > 0:
            e_accel = self.proj_A(f_accel)
            e_gyro = self.proj_G(f_gyro)
        else:
            e_accel = f_accel
            e_gyro = f_gyro
        # normalization
        e_accel = tf.math.l2_normalize(e_accel, axis=1)
        e_gyro = tf.math.l2_normalize(e_gyro, axis=1)
        # compute
        logits = tf.matmul(e_accel, e_gyro, transpose_b=True) * tf.exp(self.temperature)
        # concatanation (and send to classifier)
        combined_features = tf.concat([f_accel, f_gyro], axis=1)
        cls_output = self.classifier(combined_features, training=training)
        
        if return_feat:
            return logits, cls_output, (f_accel, f_gyro)
        
        return logits, cls_output
    
    def freeze_enc(self) :
        self.enc_A.trainable = False # 처음에는 freezing
        self.enc_G.trainable = False
        if self.proj_dim > 0: #proj_dim <= 0 --> 굳이 proj_A할 필요 없음 (기존에도)
            self.proj_A.trainable = False 
            self.proj_G.trainable = False

class CAGE_EarlyFusion(Model) :
    def __init__(self, n_feat, n_cls):
        super(CAGE_EarlyFusion, self).__init__()
        self.encoder = Encoder(n_feat, 128)
        self.classifier = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(n_cls)
        ])
        
    def call(self, x, return_feat=False, training=False):
        out = self.encoder(x, training=training)
        out = self.classifier(out, training=training)
        return out

def init_weights_orthogonal() :
    initializer = tf.keras.initializers.Orthogonal()
    return initializer

# if __name__ == "__main__":
#     model = CAGE_EarlyFusion(3, 6)
#     print(sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables))

# if __name__ == "__main__":
#     model = CAGE(3, 6, 64)
#     print(sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables))