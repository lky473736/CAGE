import tensorflow as tf
from tensorflow.keras import layers, Model

   
'''
    this is AE (autoencoder) with conv layer
    so it must looks like encoder -> latent (space) -> decoder
'''
# INVALID_ARGUMENT: Only one input size may be -1, not both 1 and 2


class VGGishBlock(layers.Layer):
    def __init__(self, filters, pool=True, last=False, stride=1):
        super(VGGishBlock, self).__init__()
        
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=(1, 3),
            strides=1,
            padding='same',
            use_bias=False
        )
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=(1, 3),
            strides=1,
            padding='same',
            use_bias=False
        )
        
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        self.pool = pool
        self.last = last
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.last:
            x = tf.tanh(x)
        else:
            x = tf.nn.relu(x)
        
        return x

class MLP_Classifier(Model):
    def __init__(self, n_cls):
        super(MLP_Classifier, self).__init__()
        
        self.dense1 = layers.Dense(2048)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.4)
        
        self.dense2 = layers.Dense(512)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.4)
        
        self.out = layers.Dense(n_cls)
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        
        return self.out(x)

class CAE(Model):
    def __init__(self, n_feat, latent_size=100):
        super(CAE, self).__init__()
        
        self.n_feat = n_feat
        
        self.up_conv1 = VGGishBlock(64) 
        self.up_conv2 = VGGishBlock(128)
        self.up_conv3 = VGGishBlock(256)
        self.up_conv4 = VGGishBlock(512)
        
        self.embedding = layers.Dense(latent_size, kernel_initializer='glorot_normal')
        self.bn1 = layers.BatchNormalization()
        
        self.de_embedding = layers.Dense(512 * n_feat, kernel_initializer='glorot_normal')
        self.bn2 = layers.BatchNormalization()
        
        self.down_conv4 = VGGishBlock(256, pool=False)
        self.down_conv3 = VGGishBlock(128, pool=False)
        self.down_conv2 = VGGishBlock(64, pool=False)
        self.down_conv1 = VGGishBlock(1, pool=False, last=True)
        
    def call(self, inputs, training=False):
        x = self.up_conv1(inputs, training=training)     # (B, 1, 6, 64)
        x = self.up_conv2(x, training=training)          # (B, 1, 6, 128)
        x = self.up_conv3(x, training=training)          # (B, 1, 6, 256)
        x = self.up_conv4(x, training=training)          # (B, 1, 6, 512)
        
        batch_size = tf.shape(inputs)[0]
        
        x = tf.reshape(x, [batch_size, -1])
        embedding = self.embedding(x)
        embedding = tf.nn.relu(self.bn1(embedding, training=training))
        
        x = self.de_embedding(embedding)
        x = tf.nn.relu(self.bn2(x, training=training))
        
        x = tf.reshape(x, [batch_size, 1, self.n_feat, 512])
        
        x = self.down_conv4(x, training=training)
        x = self.down_conv3(x, training=training)
        x = self.down_conv2(x, training=training)
        x = self.down_conv1(x, training=training)
        
        return x, embedding
    
# class CAE(Model):
#     def __init__(self, n_feat, latent_size=100):
#         super(CAE, self).__init__()
        
#         '''
#             this is AE (autoencoder) with conv layer
#             so it must looks like encoder -> latent (space) -> decoder
#         '''
#         # INVALID_ARGUMENT: Only one input size may be -1, not both 1 and 2
        
#         self.n_feat = n_feat # need reshape
#         # encoder
#         self.up_conv_1 = Vggish(64)
#         self.up_conv_2 = Vggish(128)
#         self.up_conv_3 = Vggish(256)
#         self.up_conv_4 = Vggish(512)
        
#         # latent space
#         self.embedding = layers.Dense(latent_size, kernel_initializer='glorot_normal')
#         self.bn_1 = layers.BatchNormalization()
        
#         self.de_embedding = layers.Dense(512 * 8 * n_feat, kernel_initializer='glorot_normal')
#         self.bn_2 = layers.BatchNormalization()
        
#         # decoder
#         self.down_conv_4 = Vggish(256, pool=False)
#         self.down_conv_3 = Vggish(128, pool=False)
#         self.down_conv_2 = Vggish(64, pool=False)
#         self.down_conv_1 = Vggish(1, pool=False, last=True)

#     def call(self, inputs, training=False):
#         # encoder
#         x = self.up_conv_1(inputs, training=training)  # (64, 1, 6, 64)
#         x = self.up_conv_2(x, training=training)       # (64, 1, 6, 128)
#         x = self.up_conv_3(x, training=training)       # (64, 1, 6, 256)
#         x = self.up_conv_4(x, training=training)       # (64, 1, 6, 512)

#         batch_size = tf.shape(x)[0]
#         x = tf.reshape(x, [batch_size, -1])           # flatten
#         embedding_out = self.embedding(x)              # latent space
#         embedding = tf.nn.relu(self.bn_1(embedding_out, training=training))

#         # decoder
#         x = self.de_embedding(embedding)
#         x = tf.nn.relu(self.bn_2(x, training=training))

#         input_height = tf.shape(inputs)[1]
#         input_width = tf.shape(inputs)[2]
#         x = tf.reshape(x, [batch_size, input_height, input_width, 512])

#         x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
#         x = self.down_conv_4(x, training=training)
#         x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
#         x = self.down_conv_3(x, training=training)
#         x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
#         x = self.down_conv_2(x, training=training)
#         x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
#         x = self.down_conv_1(x, training=training)

#         return x, embedding_out
