import sys
assert sys.version_info >= (3, 6), "Sonnet 2 requires Python >=3.6"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tree
import pandas as pd
import flex
import fusion

try:
  import sonnet.v2 as snt
  tf.enable_v2_behavior()
except ImportError:
  import sonnet as snt

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

class Encoder(snt.Module):
    def __init__(self, hdims, latent_dim, name=None):
        super(Encoder, self).__init__(name)
        
        self.hidden = []
        for idx, hdim in enumerate(hdims):
            layer_name = 'hidden{}'.format(idx)
            self.hidden.append(snt.Linear(hdim, name=layer_name))
            
        self.z_mean = snt.Linear(latent_dim, name='z_mean')
        self.z_log_var = snt.Linear(latent_dim, name='z_log_var')
        
    def __call__(self, x):
        
        # Run the input through the encoder network
        for hidden_layer in self.hidden:
            x = tf.nn.relu(hidden_layer(x))
        
        # Use the reparameterization trick to calculate the latent variables
        out_mean = self.z_mean(x)
        out_log_var = self.z_log_var(x)
        
        eps = tf.random.normal(out_mean.shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
        out_z = eps*tf.exp(0.5*out_log_var) + out_mean
        
        return {
            'mean': out_mean,
            'log_var': out_log_var,
            'z': out_z
        }

class Decoder(snt.Module):
    def __init__(self, hdims, vdim, name=None):
        super(Decoder, self).__init__(name)
                
        self.hidden = []
        for idx, hdim in enumerate(hdims):
            layer_name = 'hidden{}'.format(idx)
            self.hidden.append(snt.Linear(hdim, name=layer_name))
            
        self.visible = snt.Linear(vdim, 'visible')
        
    def __call__(self, x, apply_sigmoid=False):
        
        for hidden_layer in self.hidden:
            x = tf.nn.relu(hidden_layer(x))
        
        if apply_sigmoid:
            output = tf.nn.sigmoid(self.visible(x))
        else:
            output = self.visible(x)
        
        return output

class VAE(snt.Module):
    def __init__(self, encoder, decoder, name=None):
        super(VAE, self).__init__(name)
        
        self.encoder = encoder
        self.decoder = decoder
                
    def __call__(self, x, beta=1):
        # Run the encoder and decoder
        encoder_output = self.encoder(x)
        x_recon = self.decoder(encoder_output['z'], False)
        
        # Compute the loss function
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x)
        log_pxz = -tf.reduce_sum(cross_ent, axis=1)
        log_pz = log_normal_pdf(encoder_output['z'], 0.0, 0.0)
        log_qzx = log_normal_pdf(encoder_output['z'], encoder_output['mean'], encoder_output['log_var'])
        x_recon_loss = -tf.reduce_mean(log_pxz)
        loss = x_recon_loss - beta*tf.reduce_mean(log_pz - log_qzx)
        
        return {
            'x_recon': x_recon,
            'x_recon_loss': x_recon_loss,
            'loss': loss
        }

    def sample(self, batch_size):
        # Sample from the prior distribution on the latent space and run through the decoder
        latent_dim = self.decoder.hidden[0].input_size
        z = tf.random.normal([batch_size, latent_dim], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
        output = tf.nn.sigmoid(self.decoder(z))
        
        # Sample from the binary distributions coming out of the decoder to get spins
        eps = tf.random.uniform(output.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
        meas_int = tf.cast(tf.math.greater_equal(output, eps), tf.int32)

        return meas_int

class CatEncoder(snt.Module):
    def __init__(self, hdims, latent_dim, depth, name=None):
        super(CatEncoder, self).__init__(name)

        self.latent_dim = latent_dim
        self.depth = depth
        
        self.hidden = []
        for idx, hdim in enumerate(hdims):
            layer_name = 'hidden{}'.format(idx)
            self.hidden.append(snt.Linear(hdim, name=layer_name))
            
        self.z_mean = snt.Linear(latent_dim, name='z_mean')
        self.z_log_var = snt.Linear(latent_dim, name='z_log_var')
        
    def __call__(self, x):

        # Perform one hot encoding of the input
        one_hot = tf.one_hot(x, self.depth)
        x = tf.reshape(one_hot, [one_hot.shape[0],-1])
        
        # Run the input through the encoder network
        for hidden_layer in self.hidden:
            x = tf.nn.relu(hidden_layer(x))
        
        # Use the reparameterization trick to calculate the latent variables
        out_mean = self.z_mean(x)
        out_log_var = self.z_log_var(x)
        
        eps = tf.random.normal(out_mean.shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
        out_z = eps*tf.exp(0.5*out_log_var) + out_mean
        
        return {
            'one_hot': one_hot,
            'mean': out_mean,
            'log_var': out_log_var,
            'z': out_z
        }

class CatDecoder(snt.Module):
    def __init__(self, hdims, vdim, depth, name=None):
        super(CatDecoder, self).__init__(name)

        self.vdim = vdim
        self.depth = depth
                
        self.hidden = []
        for idx, hdim in enumerate(hdims):
            layer_name = 'hidden{}'.format(idx)
            self.hidden.append(snt.Linear(hdim, name=layer_name))
            
        self.visible = []
        for idx in range(vdim):
            layer_name = 'visible{}'.format(idx)
            self.visible.append(snt.Linear(depth, layer_name))
        
    def __call__(self, x):
        
        for hidden_layer in self.hidden:
            x = tf.nn.relu(hidden_layer(x))
        
        vis_stack = []
        for idx in range(self.vdim):
            vis_stack.append(tf.nn.softmax(self.visible[idx](x)))
        output = tf.stack(vis_stack, axis=1)
        
        return {
            'x_recon': output
        }

class CatVAE(snt.Module):
    def __init__(self, encoder, decoder, name=None):
        super(CatVAE, self).__init__(name)
        
        self.encoder = encoder
        self.decoder = decoder
                
    def __call__(self, x):
        # Run the encoder and decoder
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['z'])
        
        # Compute the loss function
        cross_ent = tf.metrics.categorical_crossentropy(encoder_output['one_hot'], decoder_output['x_recon'])
        log_pxz = -tf.reduce_sum(cross_ent, axis=1)
        log_pz = log_normal_pdf(encoder_output['z'], 0.0, 0.0)
        log_qzx = log_normal_pdf(encoder_output['z'], encoder_output['mean'], encoder_output['log_var'])
        x_recon_loss = -tf.reduce_mean(log_pxz)
        loss = x_recon_loss - tf.reduce_mean(log_pz - log_qzx)
        
        return {
            'x_recon': decoder_output['x_recon'],
            'x_recon_loss': x_recon_loss,
            'loss': loss
        }

    def sample(self, batch_size):
        # Sample in the CatVAE latent space using the prior and decode to get distributions by site
        latent_dim = self.encoder.latent_dim
        z = tf.random.normal([batch_size, latent_dim], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
        output = self.decoder(z)
        vdim = output['x_recon'].shape[1]
        
        # Sample from the resulting distributions
        probs = tf.reshape(output['x_recon'], [-1, output['x_recon'].shape[-1]])
        samples = tf.reshape(tf.random.categorical(tf.math.log(probs), 1), [batch_size, vdim])

        return samples

class VQEncoder(snt.Module):
    def __init__(self, hdims, latent_dim, embedding_dim, name=None):
        super(VQEncoder, self).__init__(name)

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        self.hidden = []
        for idx, hdim in enumerate(hdims):
            layer_name = 'hidden{}'.format(idx)
            self.hidden.append(snt.Linear(hdim, name=layer_name))
        
        self.z_e = snt.Linear(latent_dim*embedding_dim, name=layer_name)
        
    def __call__(self, x):
        
        # Run the input through the encoder network
        for hidden_layer in self.hidden:
            x = tf.nn.relu(hidden_layer(x))
        
        out_z_e = tf.reshape(self.z_e(x),[-1, self.latent_dim, self.embedding_dim])
        
        return {
            'z_e': out_z_e
        }

class VQDecoder(snt.Module):
    def __init__(self, hdims, vdim, name=None):
        super(VQDecoder, self).__init__(name)
                
        self.hidden = []
        for idx, hdim in enumerate(hdims):
            layer_name = 'hidden{}'.format(idx)
            self.hidden.append(snt.Linear(hdim, name=layer_name))
            
        self.visible = snt.Linear(vdim, 'visible')
        
    def __call__(self, x, apply_sigmoid=False):
        
        latent_dim = x.shape[1]
        embedding_dim = x.shape[2]
        x = tf.reshape(x, [-1, latent_dim*embedding_dim])
        
        for hidden_layer in self.hidden:
            x = tf.nn.relu(hidden_layer(x))
        
        if apply_sigmoid:
            output = tf.nn.sigmoid(self.visible(x))
        else:
            output = self.visible(x)
        
        return output

class VQVAE(snt.Module):
    def __init__(self, encoder, quantizer, decoder, name=None):
        super(VQVAE, self).__init__(name)
        
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
    
    def __call__(self, x, is_training):
        # Run the encoder and decoder
        encoder_output = self.encoder(x)
        quant_output = self.quantizer(encoder_output['z_e'], is_training)
        x_recon = self.decoder(quant_output['quantize'], False)
        
        # Compute the loss function
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x)
        log_pxz = -tf.reduce_sum(cross_ent, axis=1)
        x_recon_loss = -tf.reduce_mean(log_pxz)
        loss = x_recon_loss + quant_output['loss']
        
        return {
            'x_recon': x_recon,
            'x_recon_loss': x_recon_loss,
            'loss': loss
        }

    def sample(self, prior, batch_size):
        # Sample from the prior
        samples = prior.sample(batch_size)

        # Get the embedding vectors from the quantizer and run them through the decoder
        embeddings = self.quantizer.quantize(samples.numpy())
        output = self.decoder(embeddings, True)
        
        # Sample from the binary distributions coming out of the decoder to get spins
        eps = tf.random.uniform(output.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
        meas_int = tf.cast(tf.math.greater_equal(output, eps), tf.int32)

        return meas_int

class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    eps = tf.random.normal(tf.shape(z_mean), mean=0.0, stddev=1.0, dtype=z_mean.dtype)
    return z_mean + eps*tf.exp(0.5*z_log_var)

def build_encoder(input_dim, hidden_dims, latent_dim):
  '''Creates a VAE encoder using the Keras functional API

  Args:
    input_dim (int): Dimension of the input data
    hidden_dims (list): List of hidden dimensions
    latent_dim (int): Dimension of the latent space

  Returns:
    encoder (keras.Model): Encoder model
  '''
  input_layer = tf.keras.layers.Input(shape=(input_dim,))
  x = input_layer

  for hidden_dim in hidden_dims:
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)

  z_mean = tf.keras.layers.Dense(latent_dim)(x)
  z_log_var = tf.keras.layers.Dense(latent_dim)(x)
  z = Sampling()([z_mean, z_log_var])

  encoder = tf.keras.Model(input_layer, [z_mean, z_log_var, z], name='encoder')
  return encoder

def build_decoder(latent_dim, hidden_dims, visible_dim, apply_sigmoid=False):
  '''Creates a VAE decoder using the Keras functional API

  Args:
    latent_dim (int): Dimension of the latent space
    hidden_dims (list): List of hidden dimensions
    visible_dim (int): Dimension of the output space

  Returns:
    decoder (keras.Model): Decoder model
  '''

  z_in = tf.keras.layers.Input(shape=(latent_dim,))
  x = z_in

  for hidden_dim in hidden_dims:
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)

  if apply_sigmoid:
    output = tf.keras.layers.Dense(visible_dim, activation='sigmoid')(x)
  else:
    output = tf.keras.layers.Dense(visible_dim)(x)

  decoder = tf.keras.Model(z_in, output, name='decoder')

  return decoder

class KVAE(tf.keras.Model):
  def __init__(self, encoder, decoder, **kwargs):
    '''Creates a VAE model using the Keras functional API
    Based in part on the Keras VAE example found at:
    https://keras.io/examples/generative/vae/

    Args:
      encoder (keras.Model): Encoder model
      decoder (keras.Model): Decoder model
    '''
    super(KVAE, self).__init__(**kwargs)

    self.encoder = encoder
    self.decoder = decoder

    # Initialize loss trackers
    self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
    self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
    self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

  @property
  def metrics(self):
    return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

  def train_step(self, data):
    with tf.GradientTape() as tape:
      # Run the data through the VAE
      z_mean, z_log_var, z = self.encoder(data)
      recon = self.decoder(z)

      # Compute the reconstruction loss
      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=data, logits=recon)
      log_pxz = -tf.reduce_sum(cross_ent, axis=1)
      recon_loss = -tf.reduce_mean(log_pxz)

      # Compute the KL loss
      log_pz = log_normal_pdf(z, 0.0, 0.0)
      log_qzx = log_normal_pdf(z, z_mean, z_log_var)
      kl_loss = tf.reduce_mean(log_qzx - log_pz)

      # Compute total loss
      total_loss = recon_loss + kl_loss

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.recon_loss_tracker.update_state(recon_loss)
    self.kl_loss_tracker.update_state(kl_loss)

    return {'loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()}

  def __call__(self, x):
      # Run the encoder and decoder
      z_mean, z_log_var, z = self.encoder(x)
      x_recon = self.decoder(z)

      return x_recon

def build_sde_encoder(input_dim, avg_size, num_features, conv_sizes, hidden_dims, latent_dim):
  '''Creates a VAE encoder using the Keras functional API

  Args:
    input_dim    (int): Dimension of the input data
    avg_size     (int): Size of time window to average over
    num_features (int): Number of voltage elements (number of qubits*elements per qubit)
    conv_sizes  (list): Sizes for convolutional layers
    hidden_dims (list): List of hidden dimensions
    latent_dim   (int): Dimension of the latent space

  Returns:
    encoder (keras.Model): Encoder model
  '''
  input_layer = tf.keras.layers.Input(shape=(input_dim, num_features))
  x = input_layer

  first = True

  if avg_size is not None:
    x = tf.keras.layers.AveragePooling2D((avg_size, 1), strides=1)(x[...,tf.newaxis])
    first = False
  else:
    avg_size = 20

  for conv_idx, conv_size in enumerate(conv_sizes):
    if first:
      x = tf.keras.layers.Conv2D(conv_size, (avg_size, num_features))(x[...,tf.newaxis])
      first = False
    else:
      if conv_idx == 0:
        x = tf.keras.layers.Conv2D(conv_size, (avg_size, num_features), strides=2)(x)
      elif conv_idx == 1:
        x = tf.keras.layers.Conv2D(conv_size, (avg_size,1))(x)
      else:
        x = tf.keras.layers.Conv2D(conv_size, (avg_size,1))(x)
    x = tf.keras.layers.AveragePooling2D((avg_size,1), strides=1)(x)

  x = tf.keras.layers.Flatten()(x)

  for hidden_dim in hidden_dims:
    x = tf.keras.layers.Dense(hidden_dim)(x)
    #x = tf.keras.layers.Dense(hidden_dim, activation='tanh')(x)

  z_mean = tf.keras.layers.Dense(latent_dim)(x)
  z_log_var = tf.keras.layers.Dense(latent_dim)(x)
  z = Sampling()([z_mean, z_log_var])

  encoder = tf.keras.Model(input_layer, [z_mean, z_log_var, z], name='encoder')
  return encoder

def build_sde_rnn_encoder(input_dim, avg_size, num_features, lstm_size, td_sizes, hidden_dims, latent_dim):
  '''Creates a VAE encoder using the Keras functional API

  Args:
    input_dim    (int): Dimension of the input data
    avg_size     (int): Size of time window to average over
    num_features (int): Number of voltage elements (number of qubits*elements per qubit)
    lstm_size    (int): LSTM size for RNN layer
    td_sizes    (list): Time distributed layer sizes to apply to RNN output
    hidden_dims (list): List of hidden dimensions
    latent_dim   (int): Dimension of the latent space

  Returns:
    encoder (keras.Model): Encoder model
  '''
  input_layer = tf.keras.layers.Input(shape=(input_dim, num_features))
  x = input_layer

  first = True

  if avg_size is not None:
    x = tf.keras.layers.AveragePooling2D((avg_size, 1), strides=1)(x[...,tf.newaxis])
    first = False
  else:
    avg_size = 20

  x = tf.keras.layers.Reshape([input_dim - avg_size + 1, num_features])(x)

  rnn_layer = tf.keras.layers.LSTM(lstm_size,
                                   batch_input_shape=(input_dim, num_features),
                                   dropout=0.0,
                                   stateful=False,
                                   return_sequences=True,
                                   name='lstm_layer')

  x = rnn_layer(x)

  for td_size in td_sizes:
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(td_size, activation='relu'))(x)

  x = tf.keras.layers.Flatten()(x)

  for hidden_dim in hidden_dims:
    x = tf.keras.layers.Dense(hidden_dim)(x)
    #x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)

  z_mean = tf.keras.layers.Dense(latent_dim)(x)
  z_log_var = tf.keras.layers.Dense(latent_dim)(x)
  z = Sampling()([z_mean, z_log_var])

  encoder = tf.keras.Model(input_layer, [z_mean, z_log_var, z], name='encoder')
  return encoder

def build_sde_rnn_decoder(latent_dim, hidden_dims, visible_dim, lstm_size, td_sizes, num_features, apply_sigmoid=False):
  '''Creates a VAE decoder using the Keras functional API

  Args:
    latent_dim     (int): Dimension of the latent space
    hidden_dims   (list): List of hidden dimensions
    visible_dim    (int): Dimension of the output space
    lstm_size      (int): LSTM size for RNN layer
    td_sizes      (list): Time distributed layer sizes to apply to RNN output
    num_features   (int): Number of voltage elements (number of qubits*elements per qubit)
    apply_sigmoid (bool): If true, sigmoid activation will be applied to the output

  Returns:
    decoder (keras.Model): Decoder model
  '''

  z_in = tf.keras.layers.Input(shape=(latent_dim,))
  x = z_in

  for hidden_dim in hidden_dims:
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
  x = tf.keras.layers.Dense(visible_dim)(x)

  rnn_layer = tf.keras.layers.LSTM(lstm_size,
                                   batch_input_shape=(visible_dim, 1),
                                   dropout=0.0,
                                   stateful=False,
                                   return_sequences=True,
                                   name='lstm_layer')

  x = rnn_layer(x[...,tf.newaxis])
  for td_size in td_sizes:
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(td_size))(x)
    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(td_size, activation='relu'))(x)

  if apply_sigmoid:
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_features, activation='sigmoid'))(x)
  else:
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_features))(x)

  decoder = tf.keras.Model(z_in, output, name='decoder')

  return decoder

def build_physical_decoder(seq_len, latent_dim, hidden_dims, phys_dim, lstm_size, rho0, params, deltat):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Input(shape=(latent_dim,)))
  for hidden_dim in hidden_dims:
    #model.add(tf.keras.layers.Dense(latent_dim, activation=lambda x: fusion.max_activation_mean0(x, max_val=12, xscale=100.0)))
    model.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
  model.add(tf.keras.layers.Dense(phys_dim))

  model.add(tf.keras.layers.RepeatVector(seq_len))

  # Add the physical layer
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  a_rnn_cell_real.trainable = False
  a_rnn_cell_imag.trainable = False
  b_rnn_cell_real.trainable = False
  b_rnn_cell_imag.trainable = False

  model.add(tf.keras.layers.RNN(flex.EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                            maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                            num_traj=1, input_param=3),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer'))
  return model

class SDEVAE(tf.keras.Model):
  def __init__(self, encoder, decoder, phys_decoder, phys_dim, **kwargs):
    '''Creates a VAE model using the Keras functional API
    Based in part on the Keras VAE example found at:
    https://keras.io/examples/generative/vae/

    Args:
      encoder      (keras.Model): Encoder model
      decoder      (keras.Model): Decoder model
      phys_decoder (keras.Model): Physical SDE solver decoder
      phys_dim     (int)        : Dimension of latent space that is physical
    '''
    super(SDEVAE, self).__init__(**kwargs)

    self.encoder = encoder
    self.decoder = decoder
    self.phys_decoder = phys_decoder
    self.phys_dim = phys_dim

    # Initialize loss trackers
    self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
    self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
    self.smooth_loss_tracker = tf.keras.metrics.Mean(name='smooth_loss')
    self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')
    self.strong_meas_loss_tracker = tf.keras.metrics.Mean(name='strong_meas_loss')

  @property
  def metrics(self):
    return [self.total_loss_tracker, self.recon_loss_tracker, self.smooth_loss_tracker, self.kl_loss_tracker]

  def comp_loss(self, data):
    x, y = data

    # Run the voltage data through the VAE
    z_mean, z_log_var, z = self.encoder(x)
    recon = self.decoder(z)

    # Predict the spin probabilities for this parameter set using the physical
    # decoder
    prob_idx1 = 4 # Z_0 up probability index
    prob_idx2 = 5 # Z_1 up probability index
    #probs = self.phys_decoder(z[:,-self.phys_dim:])
    probs = self.phys_decoder(z)

    # Compute the smoothing and reconstruction losses
    #smooth_loss = tf.cast(tf.reduce_mean(tf.keras.metrics.mean_squared_error(recon, tf.stop_gradient(probs[:,:,prob_idx1]))), tf.float32)
    smooth_loss = tf.cast(tf.reduce_mean(tf.keras.metrics.mean_squared_error(recon[...,0], probs[:,:,prob_idx1]) + 
                                         tf.keras.metrics.mean_squared_error(recon[...,1], probs[:,:,prob_idx2])), tf.float32)
    stride = 64
    recon_subsamp = tf.concat([recon[:,::stride,...], recon[:,-1,tf.newaxis,...]], axis=1)
    recon_loss = tf.cast(tf.reduce_mean(tf.keras.metrics.mean_squared_error(recon_subsamp[...,0], y[:,:,prob_idx1]) + 
                                        tf.keras.metrics.mean_squared_error(recon_subsamp[...,1], y[:,:,prob_idx2])), tf.float32)

    #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=data, logits=recon)
    #log_pxz = -tf.reduce_sum(cross_ent, axis=1)
    #recon_loss = -tf.reduce_mean(log_pxz)

    # Compute the KL loss
    log_pz = log_normal_pdf(z, 0.0, 0.0)
    log_qzx = log_normal_pdf(z, z_mean, z_log_var)
    kl_loss = tf.reduce_mean(log_qzx - log_pz)

    # Compute the strong measurement loss
    #strong_meas_loss = fusion.fusion_mse_loss_2d(y, probs)
    strong_meas_loss = fusion.fusion_mse_loss_subsamp(y, probs)

    # Compute total loss
    #total_loss = recon_loss + kl_loss + strong_meas_loss
    total_loss = recon_loss + smooth_loss + strong_meas_loss

    return total_loss, recon_loss, smooth_loss, kl_loss, strong_meas_loss

  def train_step(self, data):
    with tf.GradientTape() as tape:
      total_loss, recon_loss, smooth_loss, kl_loss, strong_meas_loss = self.comp_loss(data)

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.recon_loss_tracker.update_state(recon_loss)
    self.smooth_loss_tracker.update_state(smooth_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    self.strong_meas_loss_tracker.update_state(strong_meas_loss)

    return {'loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'smooth_loss': self.smooth_loss_tracker.result(),
            #'kl_loss': self.kl_loss_tracker.result(),
            'strong_meas_loss': self.strong_meas_loss_tracker.result()}

  def __call__(self, x):
      # Run the encoder and decoder
      z_mean, z_log_var, z = self.encoder(x)
      x_recon = self.decoder(z)
      probs = self.phys_decoder(z)

      return z_mean, z_log_var, z, x_recon, probs