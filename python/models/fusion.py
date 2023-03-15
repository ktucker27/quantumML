import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import os
import sys
import math

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'sdes'))

import sde_solve
import sde_systems

def run_model(rho0, params, num_traj, mint, maxt, deltat=2**(-8), comp_iq=True):
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[1.0,0],[0,0]], dtype=tf.complex128), [num_traj,4,1])
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[0.5,0.5],[0.5,0.5]], dtype=tf.complex128), [num_traj,4,1])
  #x0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([rho0[0,0],rho0[0,1],rho0[1,1]], dtype=tf.complex128), [num_traj,3,1])
  x0 = sde_systems.wrap_rho_to_x(rho0, 2)

  d = 3
  m = 2
  p = 10

  a = sde_systems.GenoisSDE.a
  b = sde_systems.GenoisSDE.b
  bp = sde_systems.GenoisSDE.bp

  #mint = 0
  #maxt = 1.0
  #deltat = 2**(-8)

  p0 = params
  #p0[1] = 0.1
  tvec = np.arange(mint,maxt,deltat)
  wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
  emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, a, b, d, m, len(params), p0, [True, True, True], create_params=False)
  #params_ten = tf.tile(params[tf.newaxis,:], multiples=[num_traj,1])
  xvec = emod(x0, num_traj, wvec, params)
  rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,3]), 2)
  rhovec = tf.reshape(rhovec, [num_traj,-1,4])

  tvec = emod.tvec
  #wvec = emod.wvec

  # Simulate the I voltage record
  if comp_iq:
    traj_sdes = sde_systems.GenoisTrajSDE(tf.transpose(rhovec, perm=[0,2,1]), deltat)
    ai = traj_sdes.mia
    bi = traj_sdes.mib
    bpi = traj_sdes.mibp
    emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, len(params), p0, [True, True, True])
    ivec = emod_i(tf.zeros(1, dtype=tf.complex128), num_traj, wvec[:,:,0,0][:,:,tf.newaxis,tf.newaxis])

    # Simulate the Q voltage record
    aq = traj_sdes.mqa
    bq = traj_sdes.mqb
    bpq = traj_sdes.mqbp
    emod_q = sde_solve.EulerMultiDModel(mint, maxt, deltat, aq, bq, 1, 1, len(params), p0, [True, True, True])
    qvec = emod_q(tf.zeros(1, dtype=tf.complex128), num_traj, wvec[:,:,1,0][:,:,tf.newaxis,tf.newaxis])
  else:
    ivec = None
    qvec = None

  return rhovec, ivec, qvec, wvec, tvec

def get_probs(rhovec):
  sx, sy, sz = sde_systems.paulis()
  xvec = sde_systems.calc_exp(rhovec,sx)
  yvec = sde_systems.calc_exp(rhovec,sy)
  zvec = sde_systems.calc_exp(rhovec,sz)

  px = 0.5*(xvec+1)
  py = 0.5*(yvec+1)
  pz = 0.5*(zvec+1)

  return tf.stack([px, 1-px, py, 1-py, pz, 1-pz], axis=2)

def run_model_2d(params, num_traj, mint=0.0, maxt=1.0, deltat=2**(-8), comp_i=True):
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[1.0,0],[0,0]], dtype=tf.complex128), [num_traj,4,1])
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[0.5,0.5],[0.5,0.5]], dtype=tf.complex128), [num_traj,4,1])
  x0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([1.0,0,0,0,0,0,0,0,0,0], dtype=tf.complex128), [num_traj,10,1])

  d = 10
  m = 2

  a = sde_systems.RabiWeakMeasSDE.a
  b = sde_systems.RabiWeakMeasSDE.b
  #bp = sde_systems.RabiWeakMeasSDE.bp

  tvec = np.arange(mint,maxt,deltat)
  wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
  emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, a, b, d, m, len(params), params, [True, True, True, True], create_params=False)

  xvec = emod(x0, num_traj, wvec, params)
  rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,10]), 4)
  rhovec = tf.reshape(rhovec, [num_traj,-1,4,4])

  tvec = emod.tvec

  # Simulate the voltage record
  if comp_i:
    traj_sdes1 = sde_systems.RabiWeakMeasTrajSDE(rhovec, deltat, 0)
    ai = traj_sdes1.mia
    bi = traj_sdes1.mib
    emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, len(params), params, [True, True, True, True])
    ivec1 = emod_i(tf.zeros(1, dtype=tf.complex128), num_traj, wvec[:,:,0,:][:,:,tf.newaxis,:])

    traj_sdes2 = sde_systems.RabiWeakMeasTrajSDE(rhovec, deltat, 1)
    ai = traj_sdes2.mia
    bi = traj_sdes2.mib
    emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, len(params), params, [True, True, True, True])
    ivec2 = emod_i(tf.zeros(1, dtype=tf.complex128), num_traj, wvec[:,:,1,:][:,:,tf.newaxis,:])

    ivec = tf.transpose(tf.concat([ivec1, ivec2], axis=1), perm=[0,2,1])
  else:
    ivec = None

  return rhovec, ivec, wvec, tvec

class EulerRNNCell(tf.keras.layers.Layer):
  ''' An RNN cell for taking a single Euler step
  '''

  def __init__(self, rho0, maxt, deltat, num_traj=1, **kwargs):
    self.rho0 = tf.reshape(rho0, [-1])
    self.maxt = maxt
    self.deltat = deltat
    self.num_traj = num_traj
    self.params = np.array([5.0265,1.1,0.36], dtype=np.double)

    self.state_size = self.rho0.shape
    self.output_size = 6

    super(EulerRNNCell, self).__init__(**kwargs)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return tf.reshape(tf.ones([batch_size,1], dtype=tf.complex128)*tf.cast(tf.constant(self.rho0), dtype=tf.complex128), [batch_size,4])

  def run_model(self, rho, params, num_traj, mint, maxt, deltat=2**(-8), comp_iq=True):
    x0 = sde_systems.wrap_rho_to_x(rho, 2)

    d = 3
    m = 2

    a = sde_systems.GenoisSDE.a
    b = sde_systems.GenoisSDE.b

    tvec = np.arange(mint,maxt,deltat)
    wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
    emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, a, b, d, m, len(params), params, [True, True, True], create_params=False)
    xvec = emod(x0, num_traj, wvec, params)
    rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,3]), 2)
    rhovec = tf.reshape(rhovec, [-1,tvec.shape[0],4])

    return rhovec

  def call(self, inputs, states):
    rho = states[0]

    #traj_inputs = tf.stack([self.params[0]*tf.ones(tf.shape(inputs), dtype=inputs.dtype), tf.pow(inputs, 2.0) + 1.0e-8, self.params[2]*tf.ones(tf.shape(inputs), dtype=inputs.dtype)], axis=1)
    traj_inputs = tf.stack([inputs + 1.0e-8, self.params[1]*tf.ones(tf.shape(inputs), dtype=inputs.dtype), self.params[2]*tf.ones(tf.shape(inputs), dtype=inputs.dtype)], axis=1)
    traj_inputs = tf.squeeze(traj_inputs)
    traj_inputs = tf.tile(traj_inputs, multiples=[self.num_traj,1])
    rho = tf.tile(rho, multiples=[self.num_traj,1])

    # Advance the state one time step
    rhovecs = self.run_model(rho, traj_inputs, num_traj=tf.shape(traj_inputs)[0], mint=0, maxt=self.maxt, deltat=self.deltat, comp_iq=False)

    # Average over trajectories
    rhovecs = tf.reduce_mean(tf.reshape(rhovecs, [self.num_traj,-1,rhovecs.shape[1],rhovecs.shape[2]]), axis=0)
    
    # Calculate probabilities
    probs = tf.math.real(get_probs(rhovecs)[:,-1,:])
    probs = tf.math.maximum(probs,0)
    probs = tf.math.minimum(probs,1.0)

    # Deal with any NaNs that may have come out of the model
    #mask = tf.math.logical_not(tf.math.is_nan(tf.reduce_max(tf.math.real(probs), axis=[1])))
    #probs = tf.boolean_mask(probs, mask)

    return tf.concat((probs, tf.cast(inputs, dtype=tf.float64)), axis=1), [rhovecs[:,-1,:]]

class MySDELayer(tf.keras.layers.Layer):
  def __init__(self, num_traj, **kwargs):
    super(MySDELayer, self).__init__(**kwargs)
    self.num_traj=num_traj

  def call(self, inputs):
    params = np.array([5.0265,0.0,0.36], dtype=np.double)

    traj_inputs = tf.tile(inputs, multiples=[self.num_traj,1])[:,0]
    traj_inputs = tf.stack([params[0]*tf.ones(tf.shape(traj_inputs), dtype=traj_inputs.dtype), traj_inputs, params[2]*tf.ones(tf.shape(traj_inputs), dtype=traj_inputs.dtype)], axis=1)
    rhovecs = run_model(traj_inputs, num_traj=tf.shape(traj_inputs)[0], comp_iq=False)[0]
    probs = get_probs(rhovecs)

    probs_by_params = tf.reshape(probs, [tf.shape(inputs)[0], self.num_traj, tf.shape(probs)[1], tf.shape(probs)[2]])
    #mask = tf.math.logical_not(tf.math.is_nan(tf.reduce_max(tf.math.real(probs_by_params), axis=[2,3])))
    #probs_by_params = tf.boolean_mask(probs_by_params, mask)

    return tf.reduce_mean(probs_by_params, axis=1)

def fusion_loss(y_true, y_pred):
    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, tf.float32)
    y_pred_ro_results = tf.cast(y_pred, tf.float32)
    
    # Alternative option using cross entropy
    cross_ent_x = K.categorical_crossentropy(y_true_ro_results[:,:,:2], y_pred_ro_results[:,:,:2], from_logits=False)
    cross_ent_y = K.categorical_crossentropy(y_true_ro_results[:,:,2:4], y_pred_ro_results[:,:,2:4], from_logits=False)
    cross_ent_z = K.categorical_crossentropy(y_true_ro_results[:,:,4:6], y_pred_ro_results[:,:,4:6], from_logits=False)

    # Compute a regularization term that penalizes large trajectory deviations
    reg_err = tf.keras.metrics.mean_squared_error(tf.math.reduce_std(y_pred_ro_results[:,:,6], axis=0), 0.0)
    reg_mult = 0.1

    return (K.mean(cross_ent_x) + K.mean(cross_ent_y) + K.mean(cross_ent_z)) + reg_mult*reg_err

def fusion_mse_loss(y_true, y_pred):
    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, tf.float32)
    y_pred_ro_results = tf.cast(y_pred, tf.float32)

    # Compute a regularization term that penalizes large trajectory deviations
    #reg_err = tf.keras.metrics.mean_squared_error(tf.math.reduce_std(y_pred_ro_results[:,:,6], axis=0), 0.0)
    #reg_mult = 0.01

    #return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results[0,...], y_pred_ro_results))
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results[...,:6], y_pred_ro_results[:,:,:6])) #+ reg_mult*reg_err

def build_fusion_model(grp_size, seq_len, num_features, lstm_size, num_params):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Input(shape=(seq_len, grp_size)))
    # Add a masking layer to handle different weak measurement sequence lengths
    #model.add(tf.keras.layers.Masking(mask_value=lmv, input_shape=(grp_size, seq_len, num_features)))
    
    # Add the LSTM layer
    # TODO - Do we need regularization parameters?
    model.add(tf.keras.layers.LSTM(lstm_size,
                                   batch_input_shape=(seq_len, num_features),
                                   dropout=0.0,
                                   stateful=False,
                                   return_sequences=False,
                                   name='lstm_layer'))
    
    # Add a dense layer for parameters
    prob_dist = tf.keras.layers.Dense(num_params, name='dense_layer')
    model.add(prob_dist)
    #model.add(tf.keras.layers.TimeDistributed(prob_dist))

    # Add the physical model layer that maps parameters to probabilities
    # The +1 on the input shape is for duration
    #model.add(MySDELayer(input_shape=(num_params + 1)))
    
    return model

def build_stacked_model(grp_size, seq_len, num_features, lstm_size, rho0, deltat, num_params):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Input(shape=(seq_len, grp_size)))
    # Add a masking layer to handle different weak measurement sequence lengths
    #model.add(tf.keras.layers.Masking(mask_value=lmv, input_shape=(grp_size, seq_len, num_features)))
    
    # Add the LSTM layer
    model.add(tf.keras.layers.LSTM(lstm_size,
                                   batch_input_shape=(seq_len, num_features),
                                   dropout=0.0,
                                   stateful=False,
                                   return_sequences=True,
                                   name='lstm_layer'))
    
    # Add a dense layer for parameters
    prob_dist = tf.keras.layers.Dense(num_params, name='dense_layer')
    model.add(prob_dist)

    # Add the physical RNN layer
    model.add(tf.keras.layers.RNN(EulerRNNCell(maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0)),
                                  stateful=False,
                                  return_sequences=True,
                                  name='physical_layer'))
    
    return model

def param_metric(y_true, y_pred):
    return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:,-1,6], y_pred[:,-1,6]))

def max_activation(x, max_val=math.sqrt(50.0)):
  return tf.keras.activations.sigmoid(x/100.0)*max_val

def build_fusion_ae_model(seq_len, num_features, encoder_sizes, num_params, rho0, deltat):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Input(shape=(seq_len)))

    for size in encoder_sizes:
      model.add(tf.keras.layers.Dense(size, activation='relu'))

    model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: max_activation(x, max_val=15)))

    model.add(tf.keras.layers.RepeatVector(seq_len))
    
    # Add the physical RNN layer
    model.add(tf.keras.layers.RNN(EulerRNNCell(maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0)),
                                  stateful=False,
                                  return_sequences=True,
                                  name='physical_layer'))
    
    return model

def compile_model(model, loss_func, optimizer='adam', metrics=[]):
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)
