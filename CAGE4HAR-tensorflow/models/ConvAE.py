import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class Vggish(layers.Layer) :
    def __init__(self, out_channels, pool=True, last=False, stride=1):
        super(Vggish, self).__init__()
        
        # 2 conv
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
        
        # batch normalization
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        
        # token variable (like flag)
        self.last = last
        self.pool = pool
        
        if pool : # if pool that is at parameters field and pool is True
            self.max_pool = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))
            
    def call(self, inputs, training=False) :
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        
        if self.last :
            x = tf.tanh(x)
        else:
            x = tf.nn.relu(x)
            
        if self.pool:
            x = self.max_pool(x)
            
        return x

class MLP_Classifier(Model) :
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
        
        # encoder
        self.up_conv_1 = Vggish(64)
        self.up_conv_2 = Vggish(128)
        self.up_conv_3 = Vggish(256)
        self.up_conv_4 = Vggish(512)
        
        # latent space
        self.flatten = layers.Flatten()
        self.embedding = layers.Dense(latent_size)
        self.de_embedding = layers.Dense(512 * 8 * n_feat)
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        
        # decoder
        self.down_conv_4 = Vggish(256, pool=False)
        self.down_conv_3 = Vggish(128, pool=False)
        self.down_conv_2 = Vggish(64, pool=False)
        self.down_conv_1 = Vggish(1, pool=False, last=True)
        
        self.n_feat = n_feat
        
    def call(self, inputs, training=False):
        # encoder
        x = self.up_conv_1(inputs, training=training)
        x = self.up_conv_2(x, training=training)
        x = self.up_conv_3(x, training=training)
        x = self.up_conv_4(x, training=training)
        
        # latent space
        rep = self.flatten(x)
        embedding_out = self.embedding(rep)
        embedding = tf.nn.relu(self.bn_1(embedding_out, training=training))
        de_embedding = tf.nn.relu(self.bn_2(self.de_embedding(embedding), training=training))
        
        # reshape
        conv_back = tf.reshape(de_embedding, [-1, 1, 8, 512])
        
        # decoder with restoring shape
        x = tf.image.resize(conv_back, [1, 16], method='nearest')
        x = self.down_conv_4(x, training=training)
        
        x = tf.image.resize(x, [1, 32], method='nearest')
        x = self.down_conv_3(x, training=training)
        
        x = tf.image.resize(x, [1, 64], method='nearest')
        x = self.down_conv_2(x, training=training)
        
        x = tf.image.resize(x, [1, 128], method='nearest')
        x = self.down_conv_1(x, training=training)
        
        return x, embedding_out

if __name__ == "__main__":
    n_feat = 6  # number of features
    model = CAE(n_feat=n_feat, latent_size=100)
    
    sample_input = tf.random.normal([1, 1, 128, n_feat])
    output, embedding = model(sample_input)
    
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Embedding shape:", embedding.shape)
    
    total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables]) # number of parameters
    print(f"Total trainable parameters: {total_params:,}")