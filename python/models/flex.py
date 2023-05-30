import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import os
import sys
import math
import fusion

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'sdes'))

import sde_solve
import sde_systems

class EulerFlexRNNCell(tf.keras.layers.Layer):
  ''' An RNN cell for taking a single Euler step with free parameters for 
  discrepancy learning
  '''

  def __init__(self, a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag, rho0, maxt, deltat, params, num_traj=1, input_param=0, **kwargs):
    self.rho0 = tf.reshape(rho0, [-1])
    self.maxt = maxt
    self.deltat = deltat
    self.num_traj = num_traj
    self.params = params
    self.input_param = input_param
    self.pdim = 4 # TODO - Remove hard-coded dimension

    self.state_size = self.rho0.shape
    self.output_size = 43

    # Setup the flex SDE functions
    self.a_rnn_cell_real = a_rnn_cell_real
    self.a_rnn_cell_imag = a_rnn_cell_imag
    self.b_rnn_cell_real = b_rnn_cell_real
    self.b_rnn_cell_imag = b_rnn_cell_imag
    a = sde_systems.RabiWeakMeasSDE.a
    b = sde_systems.RabiWeakMeasSDE.b
    self.flex = sde_systems.FlexSDE(a, b, self.a_rnn_cell_real, self.a_rnn_cell_imag, self.b_rnn_cell_real, self.b_rnn_cell_imag)

    super(EulerFlexRNNCell, self).__init__(**kwargs)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    self.flex.init_states(batch_size*self.num_traj)
    return tf.reshape(tf.ones([batch_size,1], dtype=tf.complex128)*tf.cast(tf.constant(self.rho0), dtype=tf.complex128), [batch_size,self.pdim,self.pdim])

  def run_model(self, rho, params, num_traj, mint, maxt, deltat=2**(-8), comp_iq=True):
    x0 = sde_systems.wrap_rho_to_x(rho, self.pdim)

    d = 10
    m = 2

    tvec = np.arange(mint,maxt,deltat)
    wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
    emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, self.flex.a, self.flex.b, d, m, params.shape[1], params, [True, True, True, True], create_params=False)
    xvec = emod(x0, num_traj, wvec, params)
    rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,10]), self.pdim)
    rhovec = tf.map_fn(lambda x: sde_systems.project_to_rho(x, self.pdim), rhovec/tf.linalg.trace(rhovec)[:,tf.newaxis,tf.newaxis])
    rhovec = tf.reshape(rhovec, [num_traj,-1,self.pdim,self.pdim])

    return rhovec

  def call(self, inputs, states):
    rho = states[0]

    for ii in range(self.params.shape[0]):
      if ii == self.input_param:
        param_inputs = inputs + 1.0e-8
      else:
        param_inputs = self.params[ii]*tf.ones(tf.shape(inputs), dtype=inputs.dtype)
      
      if ii == 0:
        traj_inputs = param_inputs
      else:
        traj_inputs = tf.concat((traj_inputs, param_inputs), axis=1)

    #traj_inputs = tf.squeeze(traj_inputs)
    traj_inputs = tf.tile(traj_inputs, multiples=[self.num_traj,1])
    rho = tf.tile(rho, multiples=[self.num_traj,1,1])

    # Advance the state one time step
    rhovecs = self.run_model(rho, traj_inputs, num_traj=tf.shape(traj_inputs)[0], mint=0, maxt=self.maxt, deltat=self.deltat, comp_iq=False)

    # Average over trajectories
    rhovecs = tf.reduce_mean(tf.reshape(rhovecs, [self.num_traj,-1,tf.shape(rhovecs)[1],tf.shape(rhovecs)[2],tf.shape(rhovecs)[3]]), axis=0)
    
    # Calculate probabilities
    probs = tf.math.real(sde_systems.get_2d_probs(rhovecs)[:,-1,:])
    #probs = tf.math.maximum(probs,0)
    #probs = tf.math.minimum(probs,1.0)

    # Deal with any NaNs that may have come out of the model
    #mask = tf.math.logical_not(tf.math.is_nan(tf.reduce_max(tf.math.real(probs), axis=[1])))
    #probs = tf.boolean_mask(probs, mask)

    return tf.concat((probs, tf.cast(inputs, dtype=tf.float64)), axis=1), [rhovecs[:,-1,:,:]]

def build_flex_model(seq_len, lstm_size, rho0, params, deltat):
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.RepeatVector(seq_len, input_shape=[1]))
    
  # Add the physical RNN layer
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  model.add(tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                 maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                 num_traj=20, input_param=3),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer'))
  
  return model

def build_full_flex_model(seq_len, num_features, grp_size, avg_size, conv_sizes, encoder_sizes, lstm_size, num_params, rho0, params, deltat):
  model = tf.keras.Sequential()

  first = True

  if avg_size is not None:
    model.add(tf.keras.layers.AveragePooling2D((avg_size, 1), strides=1, input_shape=(seq_len, num_features, grp_size)))
    first = False
  else:
    avg_size = 20
  
  for conv_idx, conv_size in enumerate(conv_sizes):
    if first:
      model.add(tf.keras.layers.Conv2D(conv_size, (avg_size, num_features), input_shape=(seq_len, num_features, grp_size)))
      first = False
    else:
      if conv_idx == 0:
        model.add(tf.keras.layers.Conv2D(conv_size, (avg_size, num_features), strides=2))
      elif conv_idx == 1:
        model.add(tf.keras.layers.Conv2D(conv_size, (avg_size,1)))
      else:
        model.add(tf.keras.layers.Conv2D(conv_size, (avg_size,1)))
    model.add(tf.keras.layers.AveragePooling2D((avg_size,1), strides=1))

  model.add(tf.keras.layers.Flatten())

  for size in encoder_sizes:
    model.add(tf.keras.layers.Dense(size, activation='relu'))

  model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.max_activation_mean0(x, max_val=12)))

  model.add(tf.keras.layers.RepeatVector(seq_len, input_shape=[1]))

  # Add the physical RNN layer
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  model.add(tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                 maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                 num_traj=20, input_param=3),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer'))
  
  # Make sure the biases are zero
  # TODO - Why is this needed?
  xdim = 10
  model.layers[-1].cell.flex.a_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  model.layers[-1].cell.flex.a_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))
  model.layers[-1].cell.flex.b_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  model.layers[-1].cell.flex.b_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))

  return model