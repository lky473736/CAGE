import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
import numpy as np

class Vggish(layers.Layer):
    def __init__(self, out_channels, pool=True, last=False, stride=1):
        super(Vggish, self).__init__()
        self.last = last
        self.pool = pool

        kernel_init = initializers.GlorotNormal() # why?
        self.conv_1 = layers.Conv2D(filters=out_channels, kernel_size=(3, 3),
                                  strides=(stride, stride), padding='same',
                                  use_bias=False, kernel_initializer=kernel_init)
        self.bn_1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(filters=out_channels, kernel_size=(3, 3),
                                  strides=(stride, stride), padding='same',
                                  use_bias=False, kernel_initializer=kernel_init)
        self.bn_2 = layers.BatchNormalization()
        if pool == True : 
            self.max_pool = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))

    def call(self, inputs, training=False) :
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        if self.last :
            x = tf.tanh(x)
        else :
            x = tf.nn.relu(x)
        if self.pool :
            x = self.max_pool(x)
        return x

class MLP_Classifier(Model) :
    def __init__(self, n_cls) :
        super(MLP_Classifier, self).__init__()
        
        self.dense1 = layers.Dense(2048, kernel_initializer='glorot_normal')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.4)
        self.dense2 = layers.Dense(512, kernel_initializer='glorot_normal')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.4)
        self.out = layers.Dense(n_cls, kernel_initializer='glorot_normal')

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
        
        '''
            this is AE (autoencoder) with conv layer
            so it must looks like encoder -> latent (space) -> decoder
        '''
        # encoder
        self.up_conv_1 = Vggish(64)
        self.up_conv_2 = Vggish(128)
        self.up_conv_3 = Vggish(256)
        self.up_conv_4 = Vggish(512)
        
        # latent space
        self.embedding = layers.Dense(latent_size, kernel_initializer='glorot_normal')
        self.bn_1 = layers.BatchNormalization()
        
        self.de_embedding = layers.Dense(512 * 8 * n_feat, kernel_initializer='glorot_normal')
        self.bn_2 = layers.BatchNormalization()
        
        # decoder
        self.down_conv_4 = Vggish(256, pool=False)
        self.down_conv_3 = Vggish(128, pool=False)
        self.down_conv_2 = Vggish(64, pool=False)
        self.down_conv_1 = Vggish(1, pool=False, last=True)

    def call(self, inputs, training=False):
        # encoder
        x = self.up_conv_1(inputs, training=training)
        x = self.up_conv_2(x, training=training)
        x = self.up_conv_3(x, training=training)
        x = self.up_conv_4(x, training=training)
        
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1]) # flatten
        # x = tf.reshape(x, [128, -1])
        embedding_out = self.embedding(x) # latent space
        embedding = tf.nn.relu(self.bn_1(embedding_out, training=training))
        
        # decoder
        x = self.de_embedding(embedding)
        x = tf.nn.relu(self.bn_2(x, training=training))
        
        x = tf.reshape(x, [batch_size, -1, -1, 512]) # <----- err solving!! (size 맞추기위해)
        x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
        x = self.down_conv_4(x, training=training)
        x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
        x = self.down_conv_3(x, training=training)
        x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
        x = self.down_conv_2(x, training=training)
        x = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
        x = self.down_conv_1(x, training=training)
        
        return x, embedding_out