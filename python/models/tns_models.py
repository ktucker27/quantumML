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

  def __init__(self, rnn_cell_real, rnn_cell_imag, mps0, mpo, maxt, deltat, params, input_param=0, exp_ops=[], **kwargs):
    self.mps0 = mps0
    self.mps = mps0.substate(range(mps0.num_sites()))
    self.mpo = mpo
    self.maxt = maxt
    self.deltat = deltat
    self.params = params
    self.input_param = input_param
    self.exp_ops = exp_ops

    self.anc_mps = [networks.MPS(mps0.tensors), networks.MPS(mps0.tensors), networks.MPS(mps0.tensors)]
    self.mps_out = [networks.MPS(mps0.tensors), networks.MPS(mps0.tensors)]

    self.anc_mpo_mps = [networks.apply_mpo(mpo, mps0)]
    [self.anc_mpo_mps.append(networks.apply_mpo(exp_op, mps0)) for exp_op in exp_ops]

    self.state_size = [tf.size(x) for x in self.mps.tensors]
    self.output_size = len(exp_ops)
    self.exp_out = tf.Variable(tf.zeros(self.output_size, dtype=tf.float32), trainable=False)

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
    return list(self.mps0.tensors)

  def call(self, inputs, states):
    for idx, ten in enumerate(states):
      self.mps.set_tensor(idx, ten, val=False)

    # Advance the state one time step
    _, _, exp_out = tns_solve.tdvp(self.mpo, self.mps, self.deltat, self.maxt, 0, False, None, self.exp_ops, self.anc_mps, self.anc_mpo_mps, self.mps_out)
    self.mps.assign(self.mps_out[-1])

    self.exp_out.assign(tf.cast(tf.math.real(exp_out[:,-1]), self.exp_out.dtype))

    return self.exp_out, self.mps.tensors
