import tensorflow as tf
import numpy as np
import os
import sys
import math

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'sdes'))

import sde_solve
import sde_systems

def run_model(params, num_traj, deltat=2**(-8), comp_iq=True):
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[1.0,0],[0,0]], dtype=tf.complex128), [num_traj,4,1])
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[0.5,0.5],[0.5,0.5]], dtype=tf.complex128), [num_traj,4,1])
  x0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([1.0,0,0], dtype=tf.complex128), [num_traj,3,1])

  d = 3
  m = 2
  p = 10

  a = sde_systems.GenoisSDE.a
  b = sde_systems.GenoisSDE.b
  bp = sde_systems.GenoisSDE.bp

  mint = 0
  maxt = 1.0
  #deltat = 2**(-8)

  p0 = params
  #p0[1] = 0.1
  tvec = np.arange(mint,maxt,deltat)
  wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
  emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, a, b, d, m, len(params), p0, [True, True, True], create_params=False)
  #params_ten = tf.tile(params[tf.newaxis,:], multiples=[num_traj,1])
  xvec = emod(x0, num_traj, wvec, params)
  rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,3]), 2)
  rhovec = tf.transpose(tf.reshape(rhovec, [num_traj,-1,4]), perm=[0,2,1])

  tvec = emod.tvec
  #wvec = emod.wvec

  # Simulate the I voltage record
  if comp_iq:
    traj_sdes = sde_systems.GenoisTrajSDE(rhovec, deltat)
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

def qubit_crossentropy_loss(y_true, y_pred):
    #num_prep_states = prep_states.shape[0]
    num_traj = 1

    # Run the physical model on the output
    sde_layer = MySDELayer(num_traj)
    y_pred_ro_results = sde_layer(tf.math.pow(y_pred, 2))
    y_pred_ro_results = tf.reduce_mean(tf.math.real(y_pred_ro_results), axis=0)

    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, y_pred_ro_results.dtype)

    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results[0,...], y_pred_ro_results))
    
    # Alternative option using cross entropy
    #cross_ent_x = K.categorical_crossentropy(y_true_ro_results[:,:,:2], y_pred_ro_results[:,:,:2], from_logits=False)
    #cross_ent_y = K.categorical_crossentropy(y_true_ro_results[:,:,2:4], y_pred_ro_results[:,:,2:4], from_logits=False)
    #cross_ent_z = K.categorical_crossentropy(y_true_ro_results[:,:,4:6], y_pred_ro_results[:,:,4:6], from_logits=False)
    #batch_size = tf.cast(K.shape(y_true_ro_results)[0], cross_ent_x.dtype)
    #return K.mean(cross_ent_x) + K.mean(cross_ent_y) + K.mean(cross_ent_z)

def build_model(grp_size, seq_len, num_features, lmv, lstm_size, num_prep_states, num_params):
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

def compile_model(model, num_prep_states, optimizer='adam'):
    model.compile(loss=qubit_crossentropy_loss, optimizer=optimizer)
