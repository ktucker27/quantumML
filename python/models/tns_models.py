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

sys.path.append(os.path.join(parent, 'tns'))

import networks
import tns_solve

class TDVPFlexRNNCell(tf.keras.layers.Layer):
  ''' An RNN cell for taking a single TDVP step with free parameters for 
  discrepancy learning
  '''

  def __init__(self, rnn_cell_real, rnn_cell_imag, mps0, maxt, deltat, params, input_param=0, exp_ops=[], **kwargs):
    self.mps = mps0
    self.maxt = maxt
    self.deltat = deltat
    self.params = params
    self.input_param = input_param
    self.exp_ops = exp_ops

    self.state_size = self.mps.size()
    self.output_size = len(exp_ops)

    self.rnn_cell_real = rnn_cell_real
    self.rnn_cell_imag = rnn_cell_imag

    super(TDVPFlexRNNCell, self).__init__(**kwargs)

  def get_exp(self):
    exp_out = np.zeros([len(self.exp_ops)], dtype=np.cdouble)
    for exp_idx in range(len(self.exp_ops)):
        mpo_ms = networks.apply_mpo(self.exp_ops[exp_idx], self.mps)
        exp_out[exp_idx] = self.mps.inner(mpo_ms)/self.mps.inner(self.mps)
    return exp_out

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
    #rhovec = tf.map_fn(lambda x: sde_systems.project_to_rho(x, self.pdim), rhovec/tf.linalg.trace(rhovec)[:,tf.newaxis,tf.newaxis])
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

    # Project onto the space of physical states
    rhovecs = rhovecs[:,-1,:,:]
    rhovecs = tf.map_fn(lambda x: sde_systems.project_to_rho(x, self.pdim), rhovecs/tf.linalg.trace(rhovecs)[:,tf.newaxis,tf.newaxis])

    # Calculate probabilities
    probs = tf.math.real(sde_systems.get_2d_probs(rhovecs[:,tf.newaxis,:,:])[:,-1,:])
    #probs = tf.math.maximum(probs,0)
    #probs = tf.math.minimum(probs,1.0)

    # Deal with any NaNs that may have come out of the model
    #mask = tf.math.logical_not(tf.math.is_nan(tf.reduce_max(tf.math.real(probs), axis=[1])))
    #probs = tf.boolean_mask(probs, mask)

    return tf.concat((probs, tf.cast(inputs, dtype=tf.float64)), axis=1), [rhovecs]