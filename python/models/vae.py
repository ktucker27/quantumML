import sys
assert sys.version_info >= (3, 6), "Sonnet 2 requires Python >=3.6"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tree
import pandas as pd

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
                
    def __call__(self, x):
        # Run the encoder and decoder
        encoder_output = self.encoder(x)
        x_recon = self.decoder(encoder_output['z'], False)
        
        # Compute the loss function
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x)
        log_pxz = -tf.reduce_sum(cross_ent, axis=1)
        log_pz = log_normal_pdf(encoder_output['z'], 0.0, 0.0)
        log_qzx = log_normal_pdf(encoder_output['z'], encoder_output['mean'], encoder_output['log_var'])
        x_recon_loss = -tf.reduce_mean(log_pxz)
        loss = x_recon_loss - tf.reduce_mean(log_pz - log_qzx)
        
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