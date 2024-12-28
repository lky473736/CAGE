import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class Vggish(layers.Layer):
    def __init__(self, out_channels, pool=True, last=False, stride=1):
        super(Vggish, self).__init__()
        
        # Convolutional layers
        self.conv_1 = layers.Conv2D(
            filters=out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding='same',
            use_bias=False
        )
        
        self.conv_2 = layers.Conv2D(
            filters=out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding='same',
            use_bias=False
        )
        
        # Batch normalization
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        
        # Flags
        self.last = last
        self.pool = pool
        
        if pool:
            self.max_pool = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))
        
        # Initialize weights using Xavier (glorot) initialization
        for layer in [self.conv_1, self.conv_2]:
            layer.build((None, None, None, out_channels))
            weights = tf.keras.initializers.GlorotNormal()(layer.kernel.shape)
            layer.kernel.assign(weights)
    
    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        
        if self.last:
            x = tf.tanh(x)
        else:
            x = tf.nn.relu(x)
            
        if self.pool:
            x = self.max_pool(x)
            
        return x

class MLP_Classifier(Model):
    def __init__(self, n_cls):
        super(MLP_Classifier, self).__init__()
        
        self.linear_1 = layers.Dense(2048)
        self.linear_2 = layers.Dense(512)
        self.out = layers.Dense(n_cls)
        
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        
        self.dropout_1 = layers.Dropout(0.4)
        self.dropout_2 = layers.Dropout(0.4)
        
    def call(self, inputs, training=False):
        x = self.linear_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout_1(x, training=training)
        
        x = self.linear_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout_2(x, training=training)
        
        return self.out(x)

class CAE(Model):
    def __init__(self, n_feat, latent_size=100):
        super(CAE, self).__init__()
        
        # Encoder
        self.up_conv_1 = Vggish(64)
        self.up_conv_2 = Vggish(128)
        self.up_conv_3 = Vggish(256)
        self.up_conv_4 = Vggish(512)
        
        # Latent space
        self.embedding = layers.Dense(latent_size)
        self.de_embedding = layers.Dense(512 * 8 * n_feat)
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        
        # Decoder
        self.down_conv_4 = Vggish(256, pool=False)
        self.down_conv_3 = Vggish(128, pool=False)
        self.down_conv_2 = Vggish(64, pool=False)
        self.down_conv_1 = Vggish(1, pool=False, last=True)
        
        self.n_feat = n_feat
        
    def call(self, inputs, training=False):
        # Expand dimensions to make it compatible with Conv2D
        x = tf.expand_dims(inputs, axis=1)
        
        # Encoder
        conv_1 = self.up_conv_1(x, training=training)
        conv_2 = self.up_conv_2(conv_1, training=training)
        conv_3 = self.up_conv_3(conv_2, training=training)
        conv_4 = self.up_conv_4(conv_3, training=training)
        
        # Flatten and encode
        rep = tf.reshape(conv_4, [tf.shape(conv_4)[0], -1])
        embedding_out = self.embedding(rep)
        embedding = tf.nn.relu(self.bn_1(embedding_out, training=training))
        
        # Decode
        de_embedding = tf.nn.relu(self.bn_2(self.de_embedding(embedding), training=training))
        conv_back = tf.reshape(de_embedding, tf.shape(conv_4))
        
        # Decoder with upsampling
        pad_4 = tf.image.resize(conv_back, [tf.shape(conv_back)[1], tf.shape(conv_back)[3] * 2], method='nearest')
        de_conv_4 = self.down_conv_4(pad_4, training=training)
        
        pad_3 = tf.image.resize(de_conv_4, [tf.shape(de_conv_4)[1], tf.shape(de_conv_4)[3] * 2], method='nearest')
        de_conv_3 = self.down_conv_3(pad_3, training=training)
        
        pad_2 = tf.image.resize(de_conv_3, [tf.shape(de_conv_3)[1], tf.shape(de_conv_3)[3] * 2], method='nearest')
        de_conv_2 = self.down_conv_2(pad_2, training=training)
        
        pad_1 = tf.image.resize(de_conv_2, [tf.shape(de_conv_2)[1], tf.shape(de_conv_2)[3] * 2], method='nearest')
        de_conv_1 = self.down_conv_1(pad_1, training=training)
        
        return de_conv_1, embedding_out

if __name__ == "__main__":
    # Test the model
    model = CAE(n_feat=6, latent_size=100)
    test_input = tf.random.normal((1, 6, 128))
    output, embedding = model(test_input)
    print("Output shape:", output.shape)
    print("Embedding shape:", embedding.shape)
    
    # Test classifier
    classifier = MLP_Classifier(n_cls=6)
    test_input = tf.random.normal((1, 100))
    output = classifier(test_input)
    print("Classifier output shape:", output.shape)