import tensorflow as tf
import numpy as np
import scipy
from tensorflow.keras import backend as K
import os
import sys
import math
import pickle

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'sdes'))

import sde_solve
import sde_systems

def lin_func(x, m, b):
  return m*x + b

def lsq_diff(x, y, x_window, fit_func):
  xdelta = int(x_window/2)
  new_x = x[xdelta:-xdelta]
  new_y = np.zeros(new_x.shape, dtype=y.dtype)
  for newidx in range(new_x.shape[0]):
    xidx = newidx+xdelta
    m = (y[xidx+xdelta] - y[xidx-xdelta])/(x[xidx+xdelta] - x[xidx-xdelta])
    b = y[xidx-xdelta]
    result = scipy.optimize.curve_fit(fit_func, x[xidx-xdelta:xidx+xdelta] - x[xidx-xdelta], y[xidx-xdelta:xidx+xdelta], [m,b])
    new_y[newidx] = result[0][0]

  return new_x, new_y

def split_data(data_x, data_y, train_frac):
    steps_per_val = int(1/(1 - train_frac))
    all_idcs = np.arange(data_x.shape[0])
    val_idcs = all_idcs[0::steps_per_val]
    train_idcs = np.delete(all_idcs, val_idcs)
    
    train_x = data_x[train_idcs, ...]
    valid_x = data_x[val_idcs, ...]
    
    train_y = data_y[train_idcs, ...]
    valid_y = data_y[val_idcs, ...]
    
    return train_x, valid_x, train_y, valid_y

def time_thin(data, frac):
    steps_per_val = int(1/(frac))
    all_idcs = np.arange(data.shape[1])
    val_idcs = all_idcs[0::steps_per_val]

    return data[:,val_idcs,...]

def sample_thin(data, frac):
    steps_per_val = int(1/(frac))
    all_idcs = np.arange(data.shape[0])
    val_idcs = all_idcs[0::steps_per_val]

    return data[val_idcs,...]

def save_model(model, modeldir):
  for idx, val in enumerate(model.trainable_variables):
    savedir = os.path.join(modeldir, f'{idx}')
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    tf.saved_model.save(val, savedir)

def load_model(model, modeldir):
  for idx, val in enumerate(model.trainable_variables):
    savedir = os.path.join(modeldir, f'{idx}')
    saved_val = tf.saved_model.load(savedir)
    val.assign(saved_val)

def analyze_hist(basedir, metric_names, hist_dir='histories'):
  basefiles = os.listdir(basedir)
  basefiles.sort()

  for basefile in [os.path.join(basedir, x) for x in basefiles]:
    if os.path.isdir(basefile):
      if hist_dir in os.listdir(basefile):
        historydir = os.path.join(basefile, hist_dir)
        print('Loading histories from', historydir)

        history_files = os.listdir(historydir)
        histories = []
        final_losses = []
        final_val_losses = []
        final_val_metric = {}
        final_test_losses = []
        final_test_metric = {}

        for idx, history_file in enumerate(history_files):
          if history_file.find('.dat') < 0:
            continue

          # Load the history
          with open(os.path.join(historydir,history_file), "rb") as file_pi:
            history = pickle.load(file_pi)

          histories += [history]

          epoch_idx = -1
          final_losses += [history['loss'][epoch_idx]]

          valid_vals = history['valid_metrics'][epoch_idx]
          test_vals = history['test_metrics'][epoch_idx]

          vlosses = []
          for d in valid_vals:
            vlosses += [d['loss']]
          final_val_losses += [np.mean(vlosses)]

          tlosses = []
          for d in test_vals:
            tlosses += [d['loss']]
          final_test_losses += [np.mean(tlosses)]

          for metric_name in metric_names:
            if idx == 0:
              final_val_metric[metric_name] = []
              final_test_metric[metric_name] = []

            vmetrics = []
            for d in valid_vals:
              vmetrics += [d[metric_name]]
            final_val_metric[metric_name] += [np.mean(vmetrics)]

            tmetrics = []
            for d in test_vals:
              tmetrics += [d[metric_name]]
            final_test_metric[metric_name] += [np.mean(tmetrics)]

        print(f'Loaded {len(histories)} histories')

        for metric_name in metric_names:
          print(metric_name + ':')

          sorted_test_metrics = np.take(final_test_metric[metric_name], np.argsort(final_val_losses))

          num_vals = sorted_test_metrics.shape[0]
          print('Percentiles:', sorted_test_metrics[::np.round(0.25*num_vals).astype(np.int32)])
          print('Min:', np.min(final_test_metric[metric_name]))
          print('Mean:', np.mean(final_test_metric[metric_name]))

def init_to_onehot(x_data, y_data):
  '''
  Encodes initial conditions as a one-hot vector

  Inputs:
  x_data - shape = [num_params, num_times, num_qubits, 1, num_init]
  y_data - shape = [num_params, num_times, num_probs, num_init]

  Outputs:
  onehot_x_data - shape = [num_init*num_params, num_times, num_qubits + num_init] 
                  such that the last num_init elements in the last dimension are
                  a one-hot vector indicating the initial state, and the first num_qubits
                  values are the voltage data. The first index is in forward
                  major-order, i.e. parameter value toggles first
  onehot_y_data - shape = [num_init*num_params, num_times, num_probs, num_init + 1]
  '''
  # Squeeze the x data
  x_data = tf.squeeze(x_data)
  num_params, num_times, num_qubits, num_init = x_data.shape
  _, _, num_probs, _ = y_data.shape

  # Reshape to group params and inits
  x_data = tf.transpose(x_data, perm=[3,0,1,2])
  x_data = tf.reshape(x_data, [num_init*num_params, num_times, num_qubits])

  y_data = tf.transpose(y_data, perm=[3,0,1,2])
  y_data = tf.reshape(y_data, [num_init*num_params, num_times, num_probs])

  # Create the one-hot encoding
  onehot = tf.one_hot(range(num_init), num_init, dtype=x_data.dtype)
  onehot = tf.repeat(onehot, repeats=num_params, axis=0)

  # Concat one-hot with data
  y_data = y_data[:,:,:,tf.newaxis]
  x_data = tf.concat([x_data, onehot[:,tf.newaxis,:]*tf.ones(x_data.shape, x_data.dtype)], axis=2)
  y_data = tf.concat([y_data, onehot[:,tf.newaxis,tf.newaxis,:]*tf.ones(y_data.shape, y_data.dtype)], axis=3)

  return x_data, y_data, num_init

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

def sample_traj_spins(traj_probs):
  '''Samples from the given probability distributions assumed to be Bernoulli
    probabilities of spin-up
    Input:
    traj_probs - Tensor of spin up probabilities

    Output:
    z_samp - tf.float64 tensor the same shape as traj_probs of ones (spin-up)
             and zeros (spin-down)
    eps    - Uniform random tensor the same shape as traj_probs used to sample spins
    '''
  eps = tf.random.uniform(traj_probs.shape, minval=0, maxval=1, dtype=tf.float64)
  z_samp = tf.cast(eps < traj_probs, tf.float64)
  return z_samp, eps

def run_model_2d(rho0, params, num_traj, mint=0.0, maxt=1.0, deltat=2**(-8), comp_i=True, sim_noise=True, start_meas=0, wvec=None):
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[1.0,0],[0,0]], dtype=tf.complex128), [num_traj,4,1])
  #rho0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([[0.5,0.5],[0.5,0.5]], dtype=tf.complex128), [num_traj,4,1])
  #x0 = tf.reshape(tf.ones([num_traj,1,1], dtype=tf.complex128)*tf.constant([1.0,0,0,0,0,0,0,0,0,0], dtype=tf.complex128), [num_traj,10,1])
  x0 = sde_systems.wrap_rho_to_x(rho0, 4)

  d = 10
  m = 2
  p = 10

  a = lambda t,x,p: sde_systems.RabiWeakMeasSDE.a(t,x,p,start_meas)
  if sim_noise:
    b = lambda t,x,p: sde_systems.RabiWeakMeasSDE.b(t,x,p,start_meas)
  else:
    b = sde_systems.ZeroSDE.b
  bp = lambda t,x,p: sde_systems.RabiWeakMeasSDE.bp(t,x,p,start_meas)

  tvec = np.arange(mint,maxt,deltat)
  if wvec is None:
    wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
  emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, a, b, d, m, params.shape[-1], params, [True, True, True, True, True], create_params=False)
  #emod = sde_solve.MilsteinModel(mint, maxt, deltat, a, b, bp, d, m, p, len(params), params, [True, True, True, True], create_params=False)

  xvec = emod(x0, num_traj, wvec, params)
  rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,10]), 4)
  rhovec = tf.reshape(rhovec, [num_traj,-1,4,4])

  tvec = emod.tvec

  # Simulate the voltage record
  if comp_i:
    traj_sdes1 = sde_systems.RabiWeakMeasTrajSDE(rhovec, deltat, 0, start_meas)
    ai = traj_sdes1.mia
    if sim_noise:
      bi = traj_sdes1.mib
    else:
      bi = traj_sdes1.mib_zeros
    emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, params.shape[-1], params, [True, True, True, True, True], create_params=False)
    ivec1 = emod_i(tf.zeros(1, dtype=tf.complex128), num_traj, wvec[:,:,0,:][:,:,tf.newaxis,:])

    traj_sdes2 = sde_systems.RabiWeakMeasTrajSDE(rhovec, deltat, 1, start_meas)
    ai = traj_sdes2.mia
    if sim_noise:
      bi = traj_sdes2.mib
    else:
      bi = traj_sdes2.mib_zeros
    emod_i = sde_solve.EulerMultiDModel(mint, maxt, deltat, ai, bi, 1, 1, params.shape[-1], params, [True, True, True, True, True], create_params=False)
    ivec2 = emod_i(tf.zeros(1, dtype=tf.complex128), num_traj, wvec[:,:,1,:][:,:,tf.newaxis,:])

    ivec = tf.transpose(tf.concat([ivec1, ivec2], axis=1), perm=[0,2,1])
  else:
    ivec = None

  return rhovec, ivec, wvec, tvec

class EulerRNNCell(tf.keras.layers.Layer):
  ''' An RNN cell for taking a single Euler step
  '''

  def __init__(self, rho0, maxt, deltat, params, num_traj=1, input_param=0, **kwargs):
    self.rho0 = tf.reshape(rho0, [-1])
    self.maxt = maxt
    self.deltat = deltat
    self.num_traj = num_traj
    self.params = params
    self.input_param = input_param

    self.state_size = self.rho0.shape
    self.output_size = 43

    super(EulerRNNCell, self).__init__(**kwargs)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return tf.reshape(tf.ones([batch_size,1], dtype=tf.complex128)*tf.cast(tf.constant(self.rho0), dtype=tf.complex128), [batch_size,4,4])

  def run_model(self, rho, params, num_traj, mint, maxt, deltat=2**(-8), comp_iq=True):
    x0 = sde_systems.wrap_rho_to_x(rho, 4)

    d = 10
    m = 2

    a = sde_systems.RabiWeakMeasSDE.a
    b = sde_systems.RabiWeakMeasSDE.b

    tvec = np.arange(mint,maxt,deltat)
    wvec = tf.cast(tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj,tvec.shape[0]-1,m,1]), dtype=x0.dtype)
    emod = sde_solve.EulerMultiDModel(mint, maxt, deltat, a, b, d, m, params.shape[1], params, [True, True, True, True], create_params=False)
    xvec = emod(x0, num_traj, wvec, params)
    rhovec = sde_systems.unwrap_x_to_rho(tf.reshape(tf.transpose(xvec, perm=[0,2,1]), [-1,10]), 4)
    rhovec = tf.reshape(rhovec, [num_traj,-1,4,4])

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

    traj_inputs = tf.squeeze(traj_inputs)
    traj_inputs = tf.tile(traj_inputs, multiples=[self.num_traj,1])
    rho = tf.tile(rho, multiples=[self.num_traj,1,1])

    # Advance the state one time step
    rhovecs = self.run_model(rho, traj_inputs, num_traj=tf.shape(traj_inputs)[0], mint=0, maxt=self.maxt, deltat=self.deltat, comp_iq=False)

    # Average over trajectories
    rhovecs = tf.reduce_mean(tf.reshape(rhovecs, [self.num_traj,-1,tf.shape(rhovecs)[1],tf.shape(rhovecs)[2],tf.shape(rhovecs)[3]]), axis=0)
    
    # Calculate probabilities
    probs = tf.math.real(sde_systems.get_2d_probs(rhovecs)[:,-1,:])
    probs = tf.math.maximum(probs,0)
    probs = tf.math.minimum(probs,1.0)

    # Deal with any NaNs that may have come out of the model
    #mask = tf.math.logical_not(tf.math.is_nan(tf.reduce_max(tf.math.real(probs), axis=[1])))
    #probs = tf.boolean_mask(probs, mask)

    return tf.concat((probs, tf.cast(inputs, dtype=tf.float64)), axis=1), [rhovecs[:,-1,:,:]]

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

def fusion_mse_loss_2d(y_true, y_pred):
    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, tf.float32)
    y_pred_ro_results = tf.cast(y_pred, tf.float32)

    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results[...,:-1], y_pred_ro_results[...,:-1]))

def fusion_mse_loss_subsamp(y_true, y_pred):
    # Evaluate the loss for each sample
    stride = tf.cast(tf.round(tf.shape(y_pred)[1]/tf.shape(y_true)[1]), tf.int32)
    y_true_ro_results = tf.cast(y_true, tf.float32)
    y_pred_ro_results = tf.cast(y_pred, tf.float32)[:,::stride,...]
    
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results[...,:-1], y_pred_ro_results[...,:-1]))

def fusion_mse_loss_single(y_true, y_pred):
    # Evaluate the loss for each sample
    stride = tf.cast(tf.round(tf.shape(y_pred)[1]/tf.shape(y_true)[1]), tf.int32)
    y_true_ro_results = tf.cast(y_true, tf.float32)
    y_pred_ro_results = tf.cast(y_pred, tf.float32)[:,::stride,...]
    
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results[...,:6], y_pred_ro_results[...,:6]))

def fusion_mse_loss_voltage_zz(y_true, y_pred):
    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, tf.float32)[...,0]
    y_pred_ro_results = tf.cast(y_pred, tf.float32)[...,0]

    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results, y_pred_ro_results))

def fusion_mse_loss_voltage_xyz(y_true, y_pred):
    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, tf.float32)[:,:,:2,:,0]
    y_pred_ro_results = tf.cast(y_pred, tf.float32)[:,:,:,0,:]

    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results, y_pred_ro_results))

def fusion_mse_loss_shuffle(y_true, y_pred):
    '''
    y_true - [group,param,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [group,param,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    # Evaluate the loss for each sample
    y_true_ro_results = tf.cast(y_true, tf.float32)[:,:,:,:2,:,0]
    y_pred_ro_results = tf.cast(y_pred, tf.float32)[:,:,:,:,0,:]

    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results, y_pred_ro_results))

def fusion_mse_loss_weakstrong(y_true, y_pred, num_strong_probs):
    '''
    y_true - [traj,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [traj,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    # Evaluate the loss for each sample
    indices = [0]
    indices += range(2,2+num_strong_probs)
    y_true_ro_results = tf.cast(y_true, tf.float32)[:,1:,:2,:,0]
    y_pred_ro_results = tf.cast(y_pred, tf.float32)[:,:-1,:,0,:]
    weak_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_ro_results, y_pred_ro_results))

    strong_true = tf.cast(y_true, tf.float32)[:,-1,:2,:,1:1+num_strong_probs]
    strong_pred = tf.transpose(tf.cast(y_pred, tf.float32)[:,-1,:,2:2+num_strong_probs,:], perm=[0,1,3,2])
    strong_loss = tf.reduce_mean(tf.square(strong_true - strong_pred))

    strong_weight = 1.0

    return weak_loss + strong_weight*strong_loss

def eval_model(model, data_x, data_y, num_steps, batch_size):
  '''
  Computes a list of loss and metric values shuffling the dataset with each step. This will
  result in values for a different set of groups with each step

  Input:
  model - Keras model to evaluate (assumed to already be compiled with loss and metrics)
  data_x, data_y - X and Y data for evaluation
  num_steps - Number of shuffles to evaluate
  batch_size - The number of dataset values per group

  Output:
  output_vals - A list of dictionaries of size num_steps that contains all loss and metric values
  '''
  # Compute the aggregate loss and metric
  test_ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))

  output_vals = []
  for step in range(num_steps):
    test_shuff = test_ds.shuffle(data_x.shape[0]).batch(batch_size)
    valdict = model.evaluate(test_shuff, batch_size=batch_size, return_dict=True)
    output_vals += [valdict]

  return output_vals

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
    return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:,-1,-1], y_pred[:,-1,-1]))

def param_metric_volt(y_true, y_pred):
    # Parameters are appended after the voltage mean and standard deviation, so start at index 2
    return tf.sqrt(tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,-1,0,2:], y_pred[:,-1,0,2:])))

def param_metric_volt_xyz(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,-1,0,0,1:], y_pred[:,-1,0,2:,0])))

def param_metric_volt_xyz_mse(y_true, y_pred):
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,-1,0,0,1:], y_pred[:,-1,0,2:,0]))

def param_metric_omega_mse(y_true, y_pred):
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,-1,0,0,-2], y_pred[:,-1,0,-2,0]))

def param_metric_eps_mse(y_true, y_pred):
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,-1,0,0,-1], y_pred[:,-1,0,-1,0]))

def param_metric_omega_trimmed_mse(y_true, y_pred):
    trim_idx = tf.cast(y_true.shape[0]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[0] - trim_idx)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(tf.gather(y_true[:,-1,0,0,-2], trim_range, axis=0), tf.gather(y_pred[:,-1,0,-2,0], trim_range, axis=0)))

def param_metric_eps_trimmed_mse(y_true, y_pred):
    trim_idx = tf.cast(y_true.shape[0]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[0] - trim_idx)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(tf.gather(y_true[:,-1,0,0,-1], trim_range, axis=0), tf.gather(y_pred[:,-1,0,-1,0], trim_range, axis=0)))

def param_metric_volt_xyz_trimmed_mse(y_true, y_pred):
    trim_idx = tf.cast(y_true.shape[0]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[0] - trim_idx)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(tf.gather(y_true[:,-1,0,0,1:], trim_range, axis=0), tf.gather(y_pred[:,-1,0,2:,0], trim_range, axis=0)))

def param_metric_shuffle_mse(y_true, y_pred):
    '''
    y_true - [group,param,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [group,param,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,:,-1,0,0,1:], y_pred[:,:,-1,0,2:,0]))

def param_metric_shuffle_trimmed_mse(y_true, y_pred):
    '''
    y_true - [group,param,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [group,param,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    trim_idx = tf.cast(y_true.shape[1]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[1] - trim_idx)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(tf.gather(y_true[:,:,-1,0,0,1:], trim_range, axis=1), tf.gather(y_pred[:,:,-1,0,2:,0], trim_range, axis=1)))

def param_metric_shuffle_omega_trimmed_mse(y_true, y_pred):
    '''
    y_true - [group,param,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [group,param,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    trim_idx = tf.cast(y_true.shape[1]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[1] - trim_idx)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(tf.gather(y_true[:,:,-1,0,0,-2], trim_range, axis=1), tf.gather(y_pred[:,:,-1,0,-2,0], trim_range, axis=1)))

def param_metric_shuffle_eps_trimmed_mse(y_true, y_pred):
    '''
    y_true - [group,param,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [group,param,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    trim_idx = tf.cast(y_true.shape[1]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[1] - trim_idx)
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(tf.gather(y_true[:,:,-1,0,0,-1], trim_range, axis=1), tf.gather(y_pred[:,:,-1,0,-1,0], trim_range, axis=1)))

def param_metric_weakstrong(y_true, y_pred, num_strong_probs):
    '''
    y_true - [traj,time,(qubit0,qubit1,meas_num0,meas_num1),meas_idx,(volt,[strong_probs],[true_params])]
    y_pred - [traj,time,qubit,(mean,std,[strong_probs],[input_params]),meas_idx]
    '''
    return tf.sqrt(tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true[:,-1,0,0,(1+num_strong_probs):], y_pred[:,-1,0,(2+num_strong_probs):,0])))

def param_loss(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true[:,0], y_pred[:,0])

def param_loss_omega_eps_shuffle(y_true, y_pred):
    '''
    y_true - [group,param,(omega,eps)]
    y_pred - [(param,group),(omega,eps)]
    '''
    y_pred = tf.reshape(y_pred, [tf.shape(y_true)[1],-1,tf.shape(y_pred)[1]])
    y_pred = tf.transpose(y_pred, [1,0,2])
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    # Evaluate the loss for each sample
    return tf.reduce_mean(tf.square(y_true - y_pred))

def param_loss_omega_trimmed_shuffle(y_true, y_pred):
    '''
    y_true - [group,param,(omega,eps)]
    y_pred - [(param,group),(omega,eps)]
    '''
    y_pred = tf.reshape(y_pred, [tf.shape(y_true)[1],-1,tf.shape(y_pred)[1]])
    y_pred = tf.transpose(y_pred, [1,0,2])
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    trim_idx = tf.cast(y_true.shape[1]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[1] - trim_idx)
    # Evaluate the loss for each sample
    return tf.reduce_mean(tf.square(tf.gather(y_true[...,0], trim_range, axis=1) - tf.gather(y_pred[...,0], trim_range, axis=1)))

def param_loss_eps_trimmed_shuffle(y_true, y_pred):
    '''
    y_true - [group,param,(omega,eps)]
    y_pred - [(param,group),(omega,eps)]
    '''
    y_pred = tf.reshape(y_pred, [tf.shape(y_true)[1],-1,tf.shape(y_pred)[1]])
    y_pred = tf.transpose(y_pred, [1,0,2])
    y_true = tf.repeat(y_true[:1,...], tf.shape(y_pred)[0], axis=0)
    trim_idx = tf.cast(y_true.shape[1]/10, tf.int32)
    trim_range = tf.range(trim_idx,y_true.shape[1] - trim_idx)
    # Evaluate the loss for each sample
    return tf.reduce_mean(tf.square(tf.gather(y_true[...,-1], trim_range, axis=1) - tf.gather(y_pred[...,-1], trim_range, axis=1)))

def param_loss_mp(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(tf.reshape(y_true,[-1]), tf.reshape(y_pred,[-1]))

def max_activation(x, max_val=math.sqrt(50.0)):
  return tf.keras.activations.sigmoid(x/100.0)*max_val

def max_activation_mean0(x, max_val=math.sqrt(50.0), xscale=100.0, offset=0.0):
  return tf.keras.activations.sigmoid(x/xscale)*max_val - 0.5*max_val + offset

def linear_activation_scaled(x, xscale=100.0):
  return x/xscale

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

def build_fusion_ae_model_2d(seq_len, num_features, avg_size, encoder_sizes, num_params, rho0, deltat):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Input(shape=(seq_len, num_features, 1)))

    if avg_size is not None:
      model.add(tf.keras.layers.AveragePooling2D((avg_size, num_features), strides=1))

    model.add(tf.keras.layers.Reshape([-1]))

    for size in encoder_sizes:
      model.add(tf.keras.layers.Dense(size, activation='relu'))

    model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: max_activation_mean0(x, max_val=6)))

    model.add(tf.keras.layers.RepeatVector(seq_len))
    
    # Add the physical RNN layer
    model.add(tf.keras.layers.RNN(EulerRNNCell(maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0)),
                                  stateful=False,
                                  return_sequences=True,
                                  name='physical_layer'))
    
    return model

def build_fusion_cnn_model(seq_len, num_features, grp_size, avg_size, conv_sizes, encoder_sizes, num_params, rho0, params, deltat):
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

    model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: max_activation_mean0(x, max_val=12)))

    model.add(tf.keras.layers.RepeatVector(seq_len))
    
    # Add the physical RNN layer
    model.add(tf.keras.layers.RNN(EulerRNNCell(maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0), params=params, num_traj=20, input_param=3),
                                  stateful=False,
                                  return_sequences=True,
                                  name='physical_layer'))
    
    return model

def build_fusion_rnn_model(seq_len, num_features, grp_size, avg_size, lstm_size, encoder_sizes, num_params, rho0, deltat):
    model = tf.keras.Sequential()

    first = True

    if avg_size is not None:
      model.add(tf.keras.layers.AveragePooling2D((avg_size, 1), strides=1, input_shape=(seq_len, num_features, grp_size)))
      first = False
    else:
      model.add(tf.keras.layers.Input(shape=(seq_len, num_features, grp_size)))

    model.add(tf.keras.layers.Reshape([seq_len - avg_size + 1, num_features]))

    model.add(tf.keras.layers.LSTM(lstm_size,
                                   batch_input_shape=(seq_len, num_features),
                                   dropout=0.0,
                                   stateful=False,
                                   return_sequences=True,
                                   name='lstm_layer'))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16, activation='relu')))

    #model.add(tf.keras.layers.Reshape([seq_len - avg_size + 1]))
    model.add(tf.keras.layers.Flatten())

    for size in encoder_sizes:
      model.add(tf.keras.layers.Dense(size, activation='relu'))

    model.add(tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: max_activation_mean0(x, max_val=12)))

    model.add(tf.keras.layers.RepeatVector(seq_len))
    
    # Add the physical RNN layer
    model.add(tf.keras.layers.RNN(EulerRNNCell(maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0)),
                                  stateful=False,
                                  return_sequences=True,
                                  name='physical_layer'))
    
    return model

def build_fusion_multicnn_model(seq_len, num_features, grp_size, num_init, avg_size, conv_sizes, encoder_sizes, combo_sizes, num_params, rho0, deltat, add_physical=False):
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features, grp_size, num_init))

    model1 = build_fusion_cnn_model(seq_len, num_features, grp_size, avg_size, conv_sizes, encoder_sizes, num_params, rho0, deltat)
    model1.pop() # Physical layer
    model1.pop() # Repeat layer
    model1.pop() # Parameter layer

    model2 = build_fusion_cnn_model(seq_len, num_features, grp_size, avg_size, conv_sizes, encoder_sizes, num_params, rho0, deltat)
    model2.pop() # Physical layer
    model2.pop() # Repeat layer
    model2.pop() # Parameter layer

    model1_out = model1(inputs[:,:,:,:,0])
    model2_out = model2(inputs[:,:,:,:,1])

    cat_out = tf.keras.layers.Concatenate(axis=1)([model1_out, model2_out])

    for size in combo_sizes:
      if size == 0:
        continue
      cat_out = tf.keras.layers.Dense(size, activation='relu')(cat_out)

    param_out = tf.keras.layers.Dense(num_params, name='param_layer', activation=lambda x: max_activation(x, max_val=4))(cat_out)

    if add_physical:
      repeat_out = tf.keras.layers.RepeatVector(seq_len)(param_out)
      
      # Add the physical RNN layer
      physical_layer = tf.keras.layers.RNN(EulerRNNCell(maxt=1.5*deltat, deltat=deltat, rho0=tf.constant(rho0)),
                                          stateful=False,
                                          return_sequences=True,
                                          name='physical_layer')
      outputs = physical_layer(repeat_out)
    else:
      outputs = param_out
    
    return tf.keras.Model(inputs, outputs)

def compile_model(model, loss_func, optimizer='adam', metrics=[]):
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)
