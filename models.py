import tensorflow as tf
import numpy as np
class SimpleEncoder(tf.keras.Model):
    def __init__(self, k, n, layers=1, filters=32, kernel_size=3, dilation_rate=1, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.k = k
        self.n = n
        
        self.conv_layers=layers
        self.filters=filters
        
        self.kernel_size=kernel_size
        self.dilation_rate=dilation_rate

        self.reshape = tf.keras.layers.Reshape((-1,k), name='e_reshape')
        self.conv_stack = [tf.keras.layers.Conv1D(self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, padding='same', activation='relu', name=f'e_conv_{i}') for i in range(self.conv_layers)]
        self.lin = tf.keras.layers.Dense(self.n, name='e_lin')
        self.norm = tf.keras.layers.BatchNormalization(scale=False, center=False, name='e_norm')
        
    def call(self, inputs, training=True):
        x = inputs
        
        x = self.reshape(x)
        for conv_layer in self.conv_stack:
            x = conv_layer(x)
        x = self.lin(x)
        x = self.norm(x, training=training)
        
        outputs = x
        
        return outputs

class SimpleDecoder(tf.keras.Model):
    def __init__(self, k, n, layers=1, filters=32, kernel_size=3, dilation_rate=1, **kwargs):
        super(SimpleDecoder, self).__init__(**kwargs)
        self.k = k
        self.n = n

        self.conv_layers = layers
        self.filters = filters

        self.kernel_size=kernel_size
        self.dilation_rate=dilation_rate

        self.conv_stack = [tf.keras.layers.Conv1D(self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate, padding='same', activation='relu', name=f'd_conv_{i}') for i in range(self.conv_layers)]
        self.lin = tf.keras.layers.Dense(self.k, activation='sigmoid', name='d_lin')
        self.flat = tf.keras.layers.Flatten(name='d_flat')

    def call(self, inputs, training=True):
        x = inputs
        for conv_layer in self.conv_stack:
            x = conv_layer(x)
        x = self.lin(x)
        x = self.flat(x)
        
        outputs = x
        
        return outputs
        

class SimpleAE(tf.keras.Model):
    def __init__(self, k, n, snr, encoder_layers, encoder_filters, decoder_layers, decoder_filters, **kwargs):
        super(SimpleAE, self).__init__(**kwargs)
        self.k = k
        self.n = n
        self.snr = snr
        
        self.encoder_conv_layers= encoder_layers
        self.encoder_filters = encoder_filters
        
        self.decoder_conv_layers= decoder_layers
        self.decoder_filters = decoder_filters
        
        self.encoder = SimpleEncoder(self.k, self.n, self.encoder_conv_layers, self.encoder_filters, name='encoder')
        self.decoder = SimpleDecoder(self.k, self.n, self.decoder_conv_layers, self.decoder_filters, name='decoder')
        
        self.ch = AWGNLayer(SNR=snr, name='ch')
        
    def call(self, inputs, training=True):
        x = inputs
        
        x = self.encoder(x, training)
        x = self.ch(x)
        x = self.decoder(x, training)
        
        outputs = x

        return x
    
    def set_snr(self, snr):
        self.ch.set_SNR(snr)

class ConcatAE(tf.keras.models.Model):
    def __init__(self, top_model, k, n, L, submodels, **kwargs):
        super(ConcatAE, self).__init__(**kwargs)
        
        self.top_model = top_model
        self.k = k
        self.n = n
        self.L = L
        
        self.submodels = submodels
        self.num_subs = len(self.submodels)
        self.k_sub1 = int(self.k//self.num_subs)
        self.k_sub2 = int(self.k - (self.num_subs-1)*self.k_sub1)
        self.n_sub = int(self.n / self.num_subs)
        
    def _call_encoders(self, b, training=True):
        
        bsub = [b[:, i*(self.k_sub1*self.L):(i+1)*(self.k_sub1*self.L), :] for i in range(self.num_subs-1)]
        bsub.append(b[:, (self.num_subs-1)*self.k_sub1*self.L:, :])

        # send each through the encoders
        x = [ms.encoder(bs, training=training) for (bs, ms) in zip(bsub, self.submodels)]
            

        # concatenate code
        x = tf.concat(x, -1)

        
        return x
    
    def _call_decoders(self, inputs, training=True):
        # TRANSFORM!
        y = inputs

        # split into smaller pieces
        ysub = [y[:, :, i*self.n_sub:(i+1)*self.n_sub] for i in range(self.num_subs)]

        # send through smaller decoders
        b_hat = [ms.decoder(ys, training=training) for (ys, ms) in zip(ysub, self.submodels)]


        # concatenate output
        b_hat = tf.concat(b_hat, -1)
        
        return b_hat
    
    def call(self, inputs, training=True):
        x = self._call_encoders(inputs, training=training)
        
        # send through channel
        y = self.top_model.ch(x)

        b_hat = self._call_decoders(y, training=training)
        
        return b_hat
        

    
class AWGNLayer(tf.keras.layers.Layer):
    def __init__(self, SNR, **kwargs):
        super(AWGNLayer, self).__init__(**kwargs)
        self.SNR = SNR
        self.N0 = 10**-(SNR/10)
                
    def call(self, inputs):
        # noise = tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=self.N0)
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=np.sqrt(self.N0))
        return inputs + noise

    def set_SNR(self, SNR):
        self.N0 = 10**-(SNR/10)
        self.SNR = SNR
