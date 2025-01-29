import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class DefaultEncoder(Model):
    def __init__(self, in_feat, out_feat, num_encoders=1, use_skip=True):
        super(DefaultEncoder, self).__init__()
        self.use_skip = use_skip
        self.num_encoders = num_encoders
        
        for i in range(num_encoders) :
            setattr(self, f'conv1_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
            setattr(self, f'maxpool1_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            setattr(self, f'conv2_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
            setattr(self, f'maxpool2_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            setattr(self, f'conv3_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same', activation='relu'))
    
    def call (self, x, training=False) :
        for i in range (self.num_encoders) :
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
    
class SEEncoder(Model):
    def __init__(self, in_feat, out_feat, num_encoders=1, use_skip=True):
        super(SEEncoder, self).__init__()
        self.use_skip = use_skip
        self.num_encoders = num_encoders
        
        self.input_proj = layers.Conv1D(out_feat, 1, padding='same')
        self.input_norm = layers.BatchNormalization()
        
        for i in range(num_encoders):
            setattr(self, f'conv1_{i}', layers.Conv1D(filters=out_feat, kernel_size=3, padding='same'))
            setattr(self, f'conv1_5_{i}', layers.Conv1D(filters=out_feat, kernel_size=5, padding='same'))
            setattr(self, f'conv1_7_{i}', layers.Conv1D(filters=out_feat, kernel_size=7, padding='same'))
            setattr(self, f'bn1_{i}', layers.BatchNormalization())
            setattr(self, f'maxpool1_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            
            setattr(self, f'conv2_{i}', layers.Conv1D(filters=out_feat*2, kernel_size=3, padding='same'))
            setattr(self, f'bn2_{i}', layers.BatchNormalization())
            setattr(self, f'maxpool2_{i}', layers.MaxPooling1D(pool_size=2, padding='same'))
            
            setattr(self, f'conv3_{i}', layers.Conv1D(filters=out_feat, kernel_size=1, padding='same'))
            setattr(self, f'bn3_{i}', layers.BatchNormalization())
            
            # Squeeze-and-Excitation block
            setattr(self, f'se_pool_{i}', layers.GlobalAveragePooling1D())
            setattr(self, f'se_dense1_{i}', layers.Dense(out_feat // 4))
            setattr(self, f'se_dense2_{i}', layers.Dense(out_feat))
        
        self.final_norm = layers.BatchNormalization()
        
    def call(self, x, training=False):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = tf.nn.relu(x)
        
        for i in range(self.num_encoders):
            if self.use_skip and i > 0:
                identity = x
            
            # Multi-scale convolution
            conv3 = getattr(self, f'conv1_{i}')(x)
            conv5 = getattr(self, f'conv1_5_{i}')(x)
            conv7 = getattr(self, f'conv1_7_{i}')(x)
            x = tf.concat([conv3, conv5, conv7], axis=-1)  
            x = getattr(self, f'bn1_{i}')(x)
            x = tf.nn.relu(x)
            x = getattr(self, f'maxpool1_{i}')(x)
            
            x = getattr(self, f'conv2_{i}')(x)
            x = getattr(self, f'bn2_{i}')(x)
            x = tf.nn.relu(x)
            x = getattr(self, f'maxpool2_{i}')(x)
            
            x = getattr(self, f'conv3_{i}')(x)
            x = getattr(self, f'bn3_{i}')(x)
            
            # Squeeze-and-Excitation
            se = getattr(self, f'se_pool_{i}')(x)
            se = getattr(self, f'se_dense1_{i}')(se)
            se = tf.nn.relu(se)
            se = getattr(self, f'se_dense2_{i}')(se)
            se = tf.nn.sigmoid(se)
            se = tf.expand_dims(se, axis=1)
            x = x * se
            
            if self.use_skip and i > 0:
                x = x + identity
            
            x = tf.nn.relu(x)
        
        x = self.final_norm(x)
        return tf.reduce_mean(x, axis=1)

# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(inputs, inputs)
#     x = layers.Dropout(dropout)(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     res = x + inputs

#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     return x + res

class TransformerEncoder(Model):
    def __init__(self, in_feat, out_feat, num_heads=8, ff_dim=None, num_blocks=3, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        if ff_dim is None:
            ff_dim = out_feat * 4
            
        self.input_proj = layers.Conv1D(out_feat, kernel_size=3, padding='same')
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)
        
        self.pos_encoding = self._create_sinusoidal_positional_encoding(1000, out_feat)
        
        self.attention_layers = []
        for _ in range(num_blocks):
            self.attention_layers.append({
                'attn': layers.MultiHeadAttention(
                    key_dim=out_feat // num_heads,
                    num_heads=num_heads,
                    dropout=dropout
                ),
                'norm1': layers.LayerNormalization(epsilon=1e-6),
                'ffn1': layers.Dense(ff_dim, activation='gelu'),
                'ffn2': layers.Dense(out_feat),
                'norm2': layers.LayerNormalization(epsilon=1e-6),
                'dropout': layers.Dropout(dropout)
            })
        
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_dense = layers.Dense(out_feat)
    
    def _create_sinusoidal_positional_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle = pos / (10000 ** ((2 * (i//2)) / d_model))
        pos_encoding = np.zeros(angle.shape)
        pos_encoding[:, 0::2] = np.sin(angle[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, x, training=False):
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:seq_len, :]
        
        for layer in self.attention_layers:
            attn_output = layer['attn'](x, x)
            attn_output = layer['dropout'](attn_output, training=training)
            x1 = layer['norm1'](x + attn_output)
            
            ffn_output = layer['ffn1'](x1)
            ffn_output = layer['dropout'](ffn_output, training=training)
            ffn_output = layer['ffn2'](ffn_output)
            x = layer['norm2'](x1 + ffn_output)
        
        x = self.final_norm(x)
        x = tf.reduce_mean(x, axis=1)  
        x = self.output_dense(x)
        return x

class UNetEncoder(Model):
    def __init__(self, in_feat, out_feat):
        super(UNetEncoder, self).__init__()
        
        self.enc1 = layers.Conv1D(out_feat//4, 3, strides=1, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')
        self.enc2 = layers.Conv1D(out_feat//2, 3, strides=1, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')
        self.enc3 = layers.Conv1D(out_feat, 3, strides=1, activation='relu', padding='same')
        
        self.up2 = layers.UpSampling1D(size=1)  
        self.dec2 = layers.Conv1D(out_feat//2, 3, strides=1, activation='relu', padding='same')
        self.up1 = layers.UpSampling1D(size=1)  
        self.dec1 = layers.Conv1D(out_feat//4, 3, strides=1, activation='relu', padding='same')
        
        self.final = layers.Conv1D(out_feat, 1, activation='relu')
    
    def call(self, x, training=False):
        e1 = self.enc1(x) 
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        
        d2 = self.up2(e3)
        d2 = tf.concat([d2, e2], axis=-1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = tf.concat([d1, e1], axis=-1)
        d1 = self.dec1(d1)
        
        x = self.final(d1)
        return tf.reduce_mean(x, axis=1)
    

class ResNetBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return tf.nn.relu(x + inputs)

class ResNetTransformerEncoder(Model):
    def __init__(self, in_feat, out_feat, num_heads=8):
        super(ResNetTransformerEncoder, self).__init__()
        self.out_feat = out_feat
        
        if num_heads is None:
            num_heads = 8
            
        # ResNet blocks
        self.input_conv = layers.Conv1D(out_feat, 1, padding='same')
        self.resnet_blocks = [ResNetBlock(out_feat) for _ in range(3)]
        
        # Transformer blocks
        self.pos_encoding = layers.Dense(out_feat)
        self.attention_layers = [
            layers.MultiHeadAttention(
                key_dim=out_feat // num_heads, 
                num_heads=num_heads,
                dropout=0.1
            ) for _ in range(2)
        ]
        ...
        
        self.layer_norms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(2)]
        self.ffn_layers = [
            [
                layers.Dense(out_feat * 4, activation='relu'),
                layers.Dense(out_feat)
            ] for _ in range(2)
        ]
        
        self.output_layer = layers.Dense(out_feat)
        
    def call(self, x, training=False):
        # Input shaping
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        
        # ResNet path with proper reshaping
        x = self.input_conv(x)  # [batch, seq_len, channels]
        x = tf.reshape(x, [-1, seq_length, self.out_feat])
        # ResNet path
        x = self.input_conv(x)
        for block in self.resnet_blocks:
            x = block(x)
            
        # Positional encoding
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoding = self.pos_encoding(tf.cast(positions, tf.float32))
        x = x + pos_encoding
        
        # Transformer path
        for i in range(len(self.attention_layers)):
            attn_output = self.attention_layers[i](x, x)
            x = self.layer_norms[i](x + attn_output)
            
            ffn_output = self.ffn_layers[i][0](x)
            ffn_output = self.ffn_layers[i][1](ffn_output)
            x = self.layer_norms[i](x + ffn_output)
            
        # Global pooling and output
        x = tf.reduce_mean(x, axis=1)
        return self.output_layer(x)
    
class SETransformerEncoder(Model):
    def __init__(self, in_feat, out_feat, num_heads=8, ff_dim=None, num_blocks=3, dropout=0.1):
        super(SETransformerEncoder, self).__init__()
        if ff_dim is None:
            ff_dim = out_feat * 4
            
        # SE-style input processing
        self.input_proj = layers.Conv1D(out_feat, 1, padding='same')
        self.input_norm = layers.BatchNormalization()
        
        # Multi-scale convolutions like SE
        self.conv_layers = [
            layers.Conv1D(out_feat, kernel_size=k, padding='same')
            for k in [3, 5, 7]
        ]
        self.conv_norm = layers.BatchNormalization()
        
        # SE blocks
        self.se_pool = layers.GlobalAveragePooling1D()
        self.se_dense1 = layers.Dense(out_feat // 4)
        self.se_dense2 = layers.Dense(out_feat)
        
        # Transformer blocks with position encoding
        self.pos_encoding = self._create_sinusoidal_positional_encoding(1000, out_feat)
        
        self.attention_blocks = []
        for _ in range(num_blocks):
            self.attention_blocks.append({
                'attn': layers.MultiHeadAttention(
                    key_dim=out_feat // num_heads,
                    num_heads=num_heads,
                    dropout=dropout
                ),
                'norm1': layers.LayerNormalization(epsilon=1e-6),
                'ffn1': layers.Dense(ff_dim, activation='gelu'),
                'ffn2': layers.Dense(out_feat),
                'norm2': layers.LayerNormalization(epsilon=1e-6),
                'dropout': layers.Dropout(dropout)
            })
        
        # Final processing
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_dense = layers.Dense(out_feat)
    
    def _create_sinusoidal_positional_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle = pos / (10000 ** ((2 * (i//2)) / d_model))
        pos_encoding = np.zeros(angle.shape)
        pos_encoding[:, 0::2] = np.sin(angle[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, x, training=False):
        # Initial processing
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Multi-scale convolution
        conv_outputs = []
        for conv in self.conv_layers:
            conv_outputs.append(conv(x))
        x = tf.concat(conv_outputs, axis=-1)
        x = self.conv_norm(x)
        x = tf.nn.relu(x)
        
        # SE block
        se = self.se_pool(x)
        se = self.se_dense1(se)
        se = tf.nn.relu(se)
        se = self.se_dense2(se)
        se = tf.nn.sigmoid(se)
        se = tf.expand_dims(se, axis=1)
        x = x * se
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:seq_len, :]
        
        # Transformer blocks
        for block in self.attention_blocks:
            # Self-attention
            attn_output = block['attn'](x, x)
            attn_output = block['dropout'](attn_output, training=training)
            x1 = block['norm1'](x + attn_output)
            
            # Feed-forward
            ffn_output = block['ffn1'](x1)
            ffn_output = block['dropout'](ffn_output, training=training)
            ffn_output = block['ffn2'](ffn_output)
            x = block['norm2'](x1 + ffn_output)
        
        # Final processing
        x = self.final_norm(x)
        x = tf.reduce_mean(x, axis=1)
        x = self.output_dense(x)
        return x

class CAGE(Model):
    def __init__(self, n_feat, n_cls, proj_dim=0, encoder_type='default', **kwargs):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        
        if encoder_type == 'default': # <------ ENCODER selection
            self.enc_A = DefaultEncoder(n_feat, 64, kwargs.get('num_encoders', 1), kwargs.get('use_skip', True))
            self.enc_G = DefaultEncoder(n_feat, 64, kwargs.get('num_encoders', 1), kwargs.get('use_skip', True))
        elif encoder_type == 'transformer':
            self.enc_A = TransformerEncoder(n_feat, 64, kwargs.get('num_heads', 8))
            self.enc_G = TransformerEncoder(n_feat, 64, kwargs.get('num_heads', 8))
        elif encoder_type == 'resnet_transformer':
            self.enc_A = ResNetTransformerEncoder(n_feat, 64, kwargs.get('num_heads', 8))
            self.enc_G = ResNetTransformerEncoder(n_feat, 64, kwargs.get('num_heads', 8))
        elif encoder_type == 'unet':
            self.enc_A = UNetEncoder(n_feat, 64)
            self.enc_G = UNetEncoder(n_feat, 64)
        elif encoder_type == 'se':  
            self.enc_A = SEEncoder(n_feat, 64, kwargs.get('num_encoders', 1), kwargs.get('use_skip', True))
            self.enc_G = SEEncoder(n_feat, 64, kwargs.get('num_encoders', 1), kwargs.get('use_skip', True))
        
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
