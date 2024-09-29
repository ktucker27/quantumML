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

  def __init__(self, a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag, rho0,
               maxt, deltat, params, num_traj=1, input_param=[0], comp_iq=False, sim_noise=False,
               start_meas=0, meas_param=-1, num_meas=1, strong_probs=[], project_rho=True, return_wvec=False, **kwargs):
    self.rho0 = tf.reshape(rho0, [-1])
    self.maxt = maxt
    self.deltat = deltat
    self.num_traj = num_traj
    self.params = params
    self.input_param = input_param
    self.meas_param = meas_param
    self.comp_iq = comp_iq
    self.sim_noise = sim_noise
    self.start_meas = start_meas
    self.strong_probs = strong_probs
    self.project_rho = project_rho
    self.pdim = 4 # TODO - Remove hard-coded dimension
    self.m = 2

    self.return_wvec = return_wvec
    self.return_density = False

    self.pre_meas_params = np.copy(params)
    self.pre_meas_params[1] = 0 # kappa
    self.pre_meas_params[2] = 0 # eta

    self.state_size = [self.rho0.shape, self.m, 1,
                       a_rnn_cell_real.state_size[0], a_rnn_cell_imag.state_size[0], a_rnn_cell_real.state_size[1], a_rnn_cell_imag.state_size[1],
                       b_rnn_cell_real.state_size[0], b_rnn_cell_imag.state_size[0], b_rnn_cell_real.state_size[1], b_rnn_cell_imag.state_size[1]]
    self.output_size = 43

    # Setup the flex SDE functions
    self.a_rnn_cell_real = a_rnn_cell_real
    self.a_rnn_cell_imag = a_rnn_cell_imag
    self.b_rnn_cell_real = b_rnn_cell_real
    self.b_rnn_cell_imag = b_rnn_cell_imag
    if project_rho:
      self.a_dense_real = tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5), bias_initializer='zeros')
      self.a_dense_imag = tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5), bias_initializer='zeros')
      self.b_dense_real = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
      self.b_dense_imag = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
    else:
      self.a_dense_real = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
      self.a_dense_imag = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
      self.b_dense_real = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
      self.b_dense_imag = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
    a = sde_systems.RabiWeakMeasSDE.a
    b = sde_systems.RabiWeakMeasSDE.b
    self.zero_b = sde_systems.ZeroSDE.b
    self.b_rnn_cell_real.trainable = False
    self.b_rnn_cell_imag.trainable = False

    self.flex = sde_systems.FlexSDE(a, b,
                                    self.a_rnn_cell_real, self.a_rnn_cell_imag, self.b_rnn_cell_real, self.b_rnn_cell_imag,
                                    self.a_dense_real, self.a_dense_imag, self.b_dense_real, self.b_dense_imag,
                                    num_meas=num_meas)

    super(EulerFlexRNNCell, self).__init__(**kwargs)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    self.a_state_real = self.a_rnn_cell_real.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[0]
    self.a_state_imag = self.a_rnn_cell_imag.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[0]
    self.a_carry_real = self.a_rnn_cell_real.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[1]
    self.a_carry_imag = self.a_rnn_cell_imag.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[1]
    self.b_state_real = self.b_rnn_cell_real.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[0]
    self.b_state_imag = self.b_rnn_cell_imag.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[0]
    self.b_carry_real = self.b_rnn_cell_real.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[1]
    self.b_carry_imag = self.b_rnn_cell_imag.get_initial_state(batch_size=self.num_traj*batch_size, dtype=tf.float32)[1]
    self.flex.set_states(self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
                         self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag)
    return [tf.reshape(tf.ones([self.num_traj*batch_size,1], dtype=tf.complex128)*tf.cast(tf.constant(self.rho0), dtype=tf.complex128), [self.num_traj*batch_size,self.pdim,self.pdim]), tf.zeros([self.num_traj*batch_size,self.m], dtype=tf.complex128), 0.0,
            self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
            self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag]

  def set_return_density(self, return_density):
    self.return_density = return_density

  def run_model(self, rho, ivec0, params, num_traj, mint, maxt, deltat=2**(-8), tten=None):
    x0 = sde_systems.wrap_rho_to_x(rho, self.pdim)

    d = 10
    m = 2

    tvec = np.arange(mint,maxt,deltat)
    wvec = tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1], dtype=tf.math.real(x0).dtype)
    wvec = tf.complex(wvec, tf.zeros_like(wvec))
    b = self.zero_b
    if self.sim_noise:
      b = self.flex.b
    
    # Set cell states
    self.flex.set_states(self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
                         self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag)
    
    emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, self.flex.a, b, d, m, params.shape[1], params, [True, True, True, True], create_params=False, tten=tten)
    xvec = emod(x0, num_traj, wvec, params)
    rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,10]), self.pdim)
    rhovec = tf.reshape(rhovec, [num_traj,-1,self.pdim,self.pdim])

    # Update cell states
    (self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
     self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag) = self.flex.get_states()

    # Simulate the voltage record
    if self.comp_iq:
      # Project onto the space of physical states before computing voltages (the initial state shouldn't need a projection so leave it out)
      rhovec_proj = tf.reshape(rhovec[:,1:,:,:], [-1,self.pdim,self.pdim])
      if self.project_rho:
        rhovec_proj = tf.map_fn(lambda x: sde_systems.project_to_rho(x, self.pdim), rhovec_proj/tf.linalg.trace(rhovec_proj)[:,tf.newaxis,tf.newaxis])
      rhovec_proj = tf.reshape(rhovec_proj, [-1,tf.shape(rhovec)[1]-1,self.pdim,self.pdim])
      rhovec = tf.concat([rhovec[:,:1,:,:], rhovec_proj], axis=1)

      traj_sdes1 = sde_systems.RabiWeakMeasTrajSDE(rhovec, deltat, 0)
      ai = traj_sdes1.mia
      if self.sim_noise:
        bi = traj_sdes1.mib
      else:
        bi = traj_sdes1.mib_zeros
      emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, params.shape[1], params, [True, True, True, True], create_params=False)
      ivec1 = emod_i(ivec0[:,0], num_traj, wvec[:,:,0,:][:,:,tf.newaxis,:])

      traj_sdes2 = sde_systems.RabiWeakMeasTrajSDE(rhovec, deltat, 1)
      ai = traj_sdes2.mia
      if self.sim_noise:
        bi = traj_sdes2.mib
      else:
        bi = traj_sdes2.mib_zeros
      emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, params.shape[1], params, [True, True, True, True], create_params=False)
      ivec2 = emod_i(ivec0[:,1], num_traj, wvec[:,:,1,:][:,:,tf.newaxis,:])

      ivec = tf.transpose(tf.concat([ivec1, ivec2], axis=1), perm=[0,2,1])
    else:
      ivec = ivec0[:,tf.newaxis,:]

    return rhovec, ivec, wvec

  def call(self, inputs, states):
    '''
    Output:
    output_tensor, states where
    if self.comp_iq:
      output_tensor - [batch_size*num_traj, m, 2 + len(self.strong_probs) + input_dim] - Second index gives the
                      feature (qubit and value), third index is (mean, stddev, [strong_probs], [input_params])
    else:
      output_tensor - [batch_size*num_traj, num_probs + input_dim] - Second index includes all strong measurement
                      probabilities followed by input parameters
    states -  [rhovecs, ivec, t] where
    rhovecs - [batch_size*num_traj, pdim, pdim] density operator at the output time
    ivec -    [batch_size*num_traj, m] voltage values at the output time
    t -       scalar output time
    '''
    rho = states[0]
    ivec = states[1]
    t = states[2]
    (self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
     self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag) = states[3:]

    p_t = self.params
    if t < self.start_meas:
      p_t = self.pre_meas_params

    for ii in range(self.params.shape[0]):
      if ii in self.input_param:
        param_idx = self.input_param.index(ii)
        param_inputs = inputs[:,param_idx:param_idx+1] + 1.0e-8
      else:
        param_inputs = tf.cast(p_t[ii], inputs.dtype)*tf.ones(tf.shape(inputs[:,:1]), dtype=inputs.dtype)

      if ii == 0:
        traj_inputs = param_inputs
      else:
        traj_inputs = tf.concat((traj_inputs, param_inputs), axis=1)

    # Append measurement types onto the end if provided
    if self.meas_param >= 0:
      traj_inputs = tf.concat([traj_inputs, inputs[:,self.meas_param:]], axis=1)

    #traj_inputs = tf.squeeze(traj_inputs)
    traj_inputs = tf.tile(traj_inputs, multiples=[self.num_traj,1])
    #rho = tf.tile(rho, multiples=[self.num_traj,1,1])
    #ivec = tf.tile(ivec, multiples=[self.num_traj,1])

    # Advance the state one time step
    tten = tf.range(t, t+1.5*self.deltat, self.deltat)
    rhovecs, ivec, wvec = self.run_model(rho, ivec, traj_inputs, num_traj=tf.shape(traj_inputs)[0], mint=0, maxt=self.maxt, deltat=self.deltat, tten=tten)
    rhovecs = rhovecs[:,-1,:,:]
    ivec = ivec[:,-1,:]
    wvec = wvec[:,-1,:,:]

    t = t + self.deltat

    # Compute strong measurement probabilities if needed
    if not self.comp_iq or len(self.strong_probs) > 0:
      # Average over trajectories
      rhovecs_avg = tf.reduce_mean(tf.reshape(rhovecs, [self.num_traj,-1,tf.shape(rhovecs)[1],tf.shape(rhovecs)[2]]), axis=0)

      # Project onto the space of physical states
      if self.project_rho:
        rhovecs_avg = tf.map_fn(lambda x: sde_systems.project_to_rho(x, self.pdim), rhovecs_avg/tf.linalg.trace(rhovecs_avg)[:,tf.newaxis,tf.newaxis])

      if self.num_traj == 1:
        rhovecs = rhovecs_avg

      # Calculate probabilities
      probs = tf.math.real(sde_systems.get_2d_probs(rhovecs_avg[:,tf.newaxis,:,:])[:,-1,:])

    # If what we want is voltage records, then we're done
    if self.comp_iq:
      ivec_traj = tf.reshape(tf.math.real(ivec), [self.num_traj,-1,tf.shape(ivec)[1]])
      ivec_mean = tf.reduce_mean(ivec_traj, axis=0)
      ivec_std = tf.math.reduce_std(ivec_traj, axis=0)
      if len(self.strong_probs) > 0:
        ivec_out = tf.concat([ivec_mean[...,tf.newaxis], ivec_std[...,tf.newaxis], tf.tile(tf.gather(probs, self.strong_probs, axis=1)[:,tf.newaxis,:], multiples=[1,2,1]), tf.cast(tf.tile(inputs[:,tf.newaxis,:], multiples=[1,2,1]), dtype=tf.float64)], axis=-1)
      else:
        ivec_out = tf.concat([ivec_mean[...,tf.newaxis], ivec_std[...,tf.newaxis], tf.cast(tf.tile(inputs[:,tf.newaxis,:], multiples=[1,2,1]), dtype=tf.float64)], axis=-1)
      
      if self.return_wvec:
        ivec_out = tf.concat([ivec_out, tf.math.real(wvec)], axis=-1)
        ivec_out = tf.concat([ivec_out, tf.math.imag(wvec)], axis=-1)
      
      if self.return_density:
        return rhovecs, [rhovecs, ivec, t, self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
                                           self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag]
      else:
        return ivec_out, [rhovecs, ivec, t, self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
                                            self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag]

    
    #probs = tf.math.maximum(probs,0)
    #probs = tf.math.minimum(probs,1.0)

    # Deal with any NaNs that may have come out of the model
    #mask = tf.math.logical_not(tf.math.is_nan(tf.reduce_max(tf.math.real(probs), axis=[1])))
    #probs = tf.boolean_mask(probs, mask)

    if self.return_density:
      return rhovecs, [rhovecs, ivec, t, self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
                                         self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag]
    return tf.concat((probs, tf.cast(inputs[:,:], dtype=tf.float64)), axis=1), [rhovecs, ivec, t, self.a_state_real, self.a_state_imag, self.a_carry_real, self.a_carry_imag,
                                                                                                  self.b_state_real, self.b_state_imag, self.b_carry_real, self.b_carry_imag]

class SDERNNCell(tf.keras.layers.Layer):
  ''' An RNN cell for taking a single Euler step with neural network drift and diffusion functions
  '''

  def __init__(self, a_model_real, a_model_imag, b_model_real, b_model_imag, output_dim, x0, maxt, deltat, d, m, params,
               num_traj=1, use_complex=False, eqns=None, use_rev_sde=False, init_from_input=False, tmax=0.0, input_param=[],
               num_params=-1, **kwargs):
    self.x0 = x0
    self.maxt = maxt
    self.deltat = deltat
    self.num_traj = num_traj
    self.params = params
    self.input_param = input_param
    self.use_complex = use_complex
    self.init_from_input = init_from_input
    self.d = d
    self.m = m

    self.state_size = [output_dim, 1]
    self.output_size = output_dim

    self.a_model_real = a_model_real
    self.a_model_imag = a_model_imag
    self.b_model_real = b_model_real
    self.b_model_imag = b_model_imag

    # Setup the NN SDE functions
    if use_rev_sde:
      if tmax == 0.0 or num_params < 0:
        raise Exception('Must provide tmax and num_params > 0 when using reverse SDE')
      self.nn_sde = sde_systems.RevSDE(eqns, self.a_model_real, self.a_model_imag, tmax - deltat, num_params)
    else:
      if eqns is None:
        a = sde_systems.ZeroSDE.a
        b = sde_systems.ZeroSDE.b
      else:
        a = eqns.a
        b = eqns.b
      self.nn_sde = sde_systems.NNSDE(self.a_model_real, self.a_model_imag, self.b_model_real, self.b_model_imag, a, b, d, m, self.use_complex)

    super(SDERNNCell, self).__init__(**kwargs)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if self.use_complex:
      return [tf.ones([self.num_traj*batch_size,1], dtype=tf.complex128)*tf.cast(tf.constant(self.x0)[tf.newaxis,:], dtype=tf.complex128), 0.0]
    else:
      return [tf.ones([self.num_traj*batch_size,1], dtype=tf.float32)*tf.cast(tf.constant(self.x0)[tf.newaxis,:], dtype=tf.float32), 0.0]
  
  def run_model(self, x0, params, num_traj, mint, maxt, deltat=2**(-8), tten=None):
    tvec = np.arange(mint,maxt,deltat)
    wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,self.m,1]), dtype=x0.dtype)
    #wvec = tf.cast(params[:,tf.newaxis,:self.m,tf.newaxis], dtype=x0.dtype)
    emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, self.nn_sde.a, self.nn_sde.b, self.d, self.m, params.shape[1]-self.m, params[:,self.m:], None, create_params=False, tten=tten)
    xvec = emod(x0, num_traj, wvec, params[:,self.m:])

    return xvec

  def call(self, inputs, states):
    x = states[0]
    t = states[1]

    if self.init_from_input:
      if t == 0:
        x = tf.cast(inputs[:,:self.output_size], x.dtype)

    if len(self.input_param) >= 0:
      for ii in range(self.params.shape[0]):
        if ii in self.input_param:
          param_idx = self.input_param.index(ii)
          param_inputs = inputs[:,(self.output_size+param_idx):(self.output_size+param_idx+1)]
        else:
          param_inputs = self.params[ii]*tf.ones([tf.shape(inputs)[0],1], dtype=inputs.dtype)
        
        if ii == 0:
          traj_inputs = param_inputs
        else:
          traj_inputs = tf.concat((traj_inputs, param_inputs), axis=1)
    else:
      traj_inputs = tf.tile(self.params[tf.newaxis,:], multiples=[tf.shape(inputs)[0],1])
    
    traj_inputs = tf.concat([tf.cast(inputs[:,:self.output_size], traj_inputs.dtype), traj_inputs], axis=1)

    # Tile the input based on the number of desired trajectories
    traj_inputs = tf.tile(traj_inputs, multiples=[self.num_traj,1])

    # Advance the state one time step
    tten = tf.range(t, t + 1.5*self.deltat, self.deltat)
    y = self.run_model(x, traj_inputs, num_traj=tf.shape(traj_inputs)[0], mint=0, maxt=self.maxt, deltat=self.deltat, tten=tten)
    y = y[...,-1]

    t = t + self.deltat

    return y, [y, t]

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
                                                 num_traj=20, input_param=[3]),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer'))
  
  return model

def build_full_flex_model(seq_len, num_features, grp_size, avg_size, conv_sizes, encoder_sizes, lstm_size, num_params,
                          rho0, params, deltat, num_traj=1, start_meas=0, comp_iq=False, meas_op=[2,2], input_params=[4],
                          max_val=12, offset=0.0, strong_probs=[], project_rho=True):
  num_meas = 3
  params = np.concatenate([params, tf.one_hot([meas_op[0]], depth=num_meas)[0,:].numpy(), tf.one_hot([meas_op[1]], depth=num_meas)[0,:].numpy()])

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

  model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.max_activation_mean0(x, max_val=max_val, xscale=100.0, offset=offset)))
  #model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.linear_activation_scaled(x, xscale=100.0)))

  assert(num_params == len(input_params))
  model.add(tf.keras.layers.RepeatVector(seq_len, input_shape=[num_params]))

  # Add the physical RNN layer
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  model.add(tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                 maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                 num_traj=num_traj, input_param=input_params, start_meas=start_meas, comp_iq=comp_iq,
                                                 num_meas=num_meas, strong_probs=strong_probs, project_rho=project_rho),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer'))
  
  # Make sure the biases are zero
  # TODO - Why is this needed?
  #xdim = 10
  #model.layers[-1].cell.flex.a_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.a_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))

  return model

def build_cnn_flex_model(seq_len, num_features, grp_size, filt_grp_size, avg_size, conv_sizes, encoder_sizes, lstm_size, num_params,
                         rho0, params, deltat, num_traj=1, start_meas=0, comp_iq=False, meas_op=[2,2], input_params=[4],
                         max_val=12, offset=0.0, xscale=100.0, strong_probs=[], project_rho=True):
  num_meas = 3
  params = np.concatenate([params, tf.one_hot([meas_op[0]], depth=num_meas)[0,:].numpy(), tf.one_hot([meas_op[1]], depth=num_meas)[0,:].numpy()])
  print('act scale avg')

  model = tf.keras.Sequential()

  first = True

  if avg_size is not None:
    model.add(tf.keras.layers.AveragePooling3D((avg_size, filt_grp_size, num_features), strides=1, input_shape=(seq_len, grp_size, num_features, 1)))
    first = False
  else:
    avg_size = 20
  
  for conv_idx, conv_size in enumerate(conv_sizes):
    if first:
      model.add(tf.keras.layers.Conv3D(conv_size, (avg_size, filt_grp_size, num_features), input_shape=(seq_len, grp_size, num_features, 1)))
      first = False
    else:
      if conv_idx == 0:
        model.add(tf.keras.layers.Conv3D(conv_size, (avg_size, filt_grp_size, 1), strides=1))
      elif conv_idx == 1:
        model.add(tf.keras.layers.Conv3D(conv_size, (avg_size, filt_grp_size, 1)))
      else:
        model.add(tf.keras.layers.Conv3D(conv_size, (avg_size, filt_grp_size, 1)))
    model.add(tf.keras.layers.AveragePooling3D((avg_size,filt_grp_size,1), strides=(1,1,1)))

  model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=2)))

  model.add(tf.keras.layers.Flatten())

  for size in encoder_sizes:
    model.add(tf.keras.layers.Dense(size, activation='leaky_relu'))

  model.add(tf.keras.layers.Dense(num_params, name='param_layer_maxact', activation=lambda x: fusion.max_activation_mean0(x, max_val=max_val, xscale=xscale, offset=offset)))
  #model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.linear_activation_scaled(x, xscale=1.0)))
  #model.add(tf.keras.layers.Dense(num_params, name='param_layer_noact'))

  assert(num_params == len(input_params))
  model.add(tf.keras.layers.RepeatVector(seq_len, input_shape=[num_params]))

  # Add the physical RNN layer
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  model.add(tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                 maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                 num_traj=num_traj, input_param=input_params, start_meas=start_meas, comp_iq=comp_iq,
                                                 num_meas=num_meas, strong_probs=strong_probs, project_rho=project_rho),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer'))
  
  # Make sure the biases are zero
  # TODO - Why is this needed?
  #xdim = 10
  #model.layers[-1].cell.flex.a_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.a_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))

  return model

def build_multimeas_flex_model(seq_len, num_features, grp_size, avg_size, conv_sizes, encoder_sizes, lstm_size,
                               num_params, rho0, params, deltat, num_traj=1, start_meas=0, comp_iq=False,
                               strong_probs=[], project_rho=True):
  num_meas = 3
  input_layer = tf.keras.layers.Input(shape=(seq_len, num_features+1, grp_size))
  x = input_layer
  meas_params0 = tf.cast(tf.one_hot(tf.cast(x[:,-1,-2,0], tf.int32), depth=num_meas), x.dtype)
  meas_params1 = tf.cast(tf.one_hot(tf.cast(x[:,-1,-1,0], tf.int32), depth=num_meas), x.dtype)

  first = True

  if avg_size is not None:
    x = tf.keras.layers.AveragePooling2D((avg_size, 1), strides=1)(x[...,:num_features,:])
    first = False
  else:
    avg_size = 20

  for conv_idx, conv_size in enumerate(conv_sizes):
    if first:
      x = tf.keras.layers.Conv2D(conv_size, (avg_size, num_features))(x[...,:num_features,:])
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

  for size in encoder_sizes:
    x = tf.keras.layers.Dense(size, activation='relu')(x)

  x = tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.max_activation_mean0(x, max_val=12, xscale=100.0))(x)
  #x = tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.max_activation_mean0(x, max_val=6, xscale=100.0))(x)
  #x = tf.keras.layers.Lambda(lambda x: x + 1)(x)

  x = tf.concat([x, meas_params0, meas_params1], axis=1)
  x = tf.keras.layers.RepeatVector(seq_len, input_shape=[num_params+1])(x)

  # Add the physical RNN layer
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  rnn_layer = tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                   maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                   num_traj=num_traj, input_param=3, start_meas=start_meas, comp_iq=comp_iq,
                                                   meas_param=num_params, num_meas=num_meas, strong_probs=strong_probs,
                                                   project_rho=project_rho),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer')
  
  # Make sure the biases are zero
  # TODO - Why is this needed?
  #xdim = 10
  #rnn_layer.cell.flex.a_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #rnn_layer.cell.flex.a_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))

  output = rnn_layer(x)

  return tf.keras.Model(input_layer, output, name='encoder')

def build_multimeas_rnn_model(seq_len, num_features, num_meas, avg_size, enc_lstm_size, dec_lstm_size, td_sizes, encoder_sizes, num_params,
                              rho0, params, deltat, num_traj=1, start_meas=0, comp_iq=False, input_params=[4],
                              max_val=12, offset=0.0, strong_probs=[], project_rho=True, strong_probs_input=False,
                              num_per_group=-1, params_per_group=-1, encoder_only=False):
  '''
  Input:
    input_tensor - [traj, time, (qubit0,qubit1,meas_num0,meas_num1), meas_idx, (volt,[strong_probs])]
    traj = (num_per_group*num_groups, params_per_group) if num_per_group > 0
  Output:
    if encoder_only:
      output_tensor - [traj, param, meas_idx]
    else:
      if comp_iq:
        output_tensor - [traj, time, m, 2 + len(self.strong_probs) + input_dim, meas_idx] - Second index gives the
                        feature (qubit and value), third index is (mean, stddev, [strong_probs], [input_params])
      else:
        output_tensor - [traj, time, num_probs + input_dim, meas_idx] - Second index includes all strong measurement
                        probabilities followed by input parameters
  traj = (num_groups, params_per_group) if num_per_group > 0
  '''
  num_strong_probs = len(strong_probs)
  num_features_in = num_features
  if num_per_group > 0:
    assert(params_per_group > 0)
    input_layer = tf.keras.layers.Input(shape=(params_per_group, seq_len, num_features+2, num_meas))
    x = input_layer
    x = tf.reduce_mean(tf.reshape(x, [num_per_group, -1, params_per_group, seq_len,num_features+2, num_meas]), axis=0)
    x = tf.reshape(tf.transpose(x, perm=[1,0,2,3,4]), [-1, seq_len, num_features+2, num_meas])
  elif num_strong_probs == 0 or not strong_probs_input:
    input_layer = tf.keras.layers.Input(shape=(seq_len, num_features+2, num_meas))
    x = input_layer
  else:
    # Add strong probability estimates as additional features
    input_layer = tf.keras.layers.Input(shape=(seq_len, num_features+2, num_meas, 1+num_strong_probs))
    strong_prob_vals = input_layer[:,-1,-1,-1,1:]
    x = tf.concat([input_layer[:,:,:num_features,:,0], tf.ones_like(input_layer[:,:,:1,:,0])*strong_prob_vals[:,tf.newaxis,:,tf.newaxis], input_layer[:,:,num_features:,:,0]], axis=2)
    num_features += num_strong_probs
    #x = input_layer[...,0]
  meas_params0 = tf.cast(tf.one_hot(tf.cast(x[:,-1,-2,:], tf.int32), depth=3), x.dtype)
  meas_params0 = tf.reshape(meas_params0, [-1,3]) # shape = [batch_size*num_meas,3]
  meas_params1 = tf.cast(tf.one_hot(tf.cast(x[:,-1,-1,:], tf.int32), depth=3), x.dtype)
  meas_params1 = tf.reshape(meas_params1, [-1,3]) # shape = [batch_size*num_meas,3]

  x = tf.keras.layers.Reshape([seq_len, num_features*num_meas,1])(x[:,:,:num_features,:])

  if avg_size is not None:
    x = tf.keras.layers.AveragePooling2D((avg_size, 1), strides=1)(x)
  else:
    avg_size = 1

  x = tf.keras.layers.Reshape([seq_len - avg_size + 1, num_features*num_meas])(x)

  enc_rnn_layer = tf.keras.layers.LSTM(enc_lstm_size,
                                       batch_input_shape=(seq_len, num_features*num_meas),
                                       dropout=0.0,
                                       stateful=False,
                                       return_sequences=True,
                                       name='lstm_layer')

  x = enc_rnn_layer(x)

  for td_size in td_sizes:
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(td_size, activation='relu'))(x)

  x = tf.keras.layers.Flatten()(x)

  for size in encoder_sizes:
    x = tf.keras.layers.Dense(size, activation='relu')(x)

  #x = tf.concat([x, strong_prob_vals], axis=1)

  assert(num_params == len(input_params))
  x = tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.max_activation_mean0(x, max_val=max_val, xscale=100.0, offset=offset))(x)
  #x = tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: fusion.max_activation_mean0(x, max_val=6, xscale=100.0))(x)
  #x = tf.keras.layers.Lambda(lambda x: x + 1)(x)

  x = tf.repeat(x, num_meas, axis=0)

  # This is the extent of the model if it is encoder only
  if encoder_only:
    # Split the measurement types back out from the batch index
    output = tf.transpose(tf.reshape(x, [-1,num_meas,num_params]), perm=[0,2,1])

    # Split the groups back out of the first index, if requested
    if num_per_group > 0:
      output = tf.reshape(output, [params_per_group,-1,num_params,num_meas])
      output = tf.transpose(output, perm=[1,0,2,3])

    return tf.keras.Model(input_layer, output, name='encoder')

  x = tf.concat([x, meas_params0, meas_params1], axis=1)
  x = tf.keras.layers.RepeatVector(seq_len, input_shape=[num_params+6])(x)

  # Add the physical RNN layer
  if project_rho:
    a_rnn_cell_real = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5), recurrent_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5), bias_initializer='zeros')
    a_rnn_cell_imag = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5), recurrent_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5), bias_initializer='zeros')
    b_rnn_cell_real = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
    b_rnn_cell_imag = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  else:
    a_rnn_cell_real = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
    a_rnn_cell_imag = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
    b_rnn_cell_real = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
    b_rnn_cell_imag = tf.keras.layers.LSTMCell(dec_lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  dec_rnn_layer = tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                       maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                       num_traj=num_traj, input_param=input_params, start_meas=start_meas, comp_iq=comp_iq,
                                                       meas_param=num_params, num_meas=num_meas, strong_probs=strong_probs,
                                                       project_rho=project_rho),
                                      stateful=False,
                                      return_sequences=True,
                                      name='physical_layer')
  
  # Make sure the biases are zero
  # TODO - Why is this needed?
  #xdim = 10
  #rnn_layer.cell.flex.a_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #rnn_layer.cell.flex.a_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))

  x = dec_rnn_layer(x)

  # Split the measurement types back out from the batch index
  # The first num_params+2 elements of the final index include mean, std, strong measurement probabilities, and
  # then the input params prior to the concatenated meas params
  num_out = 2 + num_strong_probs + num_params
  output = tf.transpose(tf.reshape(x[...,:num_out], [-1,num_meas,seq_len,num_features_in,num_out]), perm=[0,2,3,4,1])

  # Split the groups back out of the first index, if requested
  if num_per_group > 0:
    output = tf.reshape(output, [params_per_group,-1,seq_len,num_features_in,num_out,num_meas])
    output = tf.transpose(output, perm=[1,0,2,3,4,5])

  return tf.keras.Model(input_layer, output, name='encoder')

def build_datagen_model(seq_len, num_features, rho0, num_params, params, deltat, num_traj=1, start_meas=0, 
                        sim_noise=True, comp_iq=True, input_params=[4], strong_probs=[], num_meas=1, meas_op=[],
                        return_wvec=False):
  input_layer = tf.keras.layers.Input(shape=(num_params))
  x = input_layer

  if len(meas_op) == 0:
    meas_params = tf.tile(tf.one_hot(tf.range(0,num_meas,1), depth=3), multiples=[tf.shape(x)[0],1])

    x = tf.repeat(x, num_meas, axis=0)
    x = tf.concat([x, meas_params, meas_params], axis=1)
  else:
    assert(len(meas_op) == 2)
    meas_params0 = tf.tile(tf.one_hot(meas_op[0], depth=3)[tf.newaxis,:], multiples=[tf.shape(x)[0],1])
    meas_params1 = tf.tile(tf.one_hot(meas_op[1], depth=3)[tf.newaxis,:], multiples=[tf.shape(x)[0],1])
    x = tf.concat([x, meas_params0, meas_params1], axis=1)

  x = tf.keras.layers.RepeatVector(seq_len, input_shape=[num_params])(x)

  # Add the physical RNN layer
  lstm_size = 10
  a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
  b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

  a_rnn_cell_real.trainable = False
  a_rnn_cell_imag.trainable = False
  b_rnn_cell_real.trainable = False
  b_rnn_cell_imag.trainable = False

  rnn_layer = tf.keras.layers.RNN(EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                                   maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params,
                                                   num_traj=num_traj, input_param=input_params, start_meas=start_meas, comp_iq=comp_iq,
                                                   sim_noise=sim_noise, meas_param=num_params, num_meas=num_meas,
                                                   strong_probs=strong_probs, project_rho=False, return_wvec=return_wvec),
                                stateful=False,
                                return_sequences=True,
                                name='physical_layer')
  
  # Make sure the biases are zero
  # TODO - Why is this needed?
  #xdim = 10
  #model.layers[-1].cell.flex.a_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.a_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_real.trainable_weights[-1].assign(tf.zeros(4*xdim))
  #model.layers[-1].cell.flex.b_cell_imag.trainable_weights[-1].assign(tf.zeros(4*xdim))

  x = rnn_layer(x)

  # Split the measurement types back out from the batch index
  # The first num_params+2 elements of the final index include mean, std, and then the input params prior to
  # the concatenated meas params, followed by the real and imaginary parts of the wvec
  num_strong_probs = len(strong_probs)
  num_out = 2 + num_strong_probs + num_params
  if return_wvec:
    output = tf.transpose(tf.reshape(tf.concat([x[...,:num_out], x[...,-2:]], axis=-1), [-1,num_meas,seq_len,num_features,num_out+2]), perm=[0,2,3,4,1])
  else:
    output = tf.transpose(tf.reshape(x[...,:num_out], [-1,num_meas,seq_len,num_features,num_out]), perm=[0,2,3,4,1])

  return tf.keras.Model(input_layer, output, name='data_gen_model')
