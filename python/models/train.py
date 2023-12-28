import numpy as np
import tensorflow as tf
import os
import sys
import pickle

import fusion
import flex

currpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currpath,'../sdes'))

import sde_systems

def load_dataset(datapath, data_group_size, clean, stride, group_size, num_train_groups, meas_op=[], debug=True):
  '''
  Loads data from the specified path and splits it into training/validation/test sets

  The validation set will be half the size of the training set, unless the training set has only one group.
  Disjoint parameters are used for the training and validation/test sets. Disjoint groups are used for the
  validation and test sets, unless there is only one validation/test group

  If only one parameter is specified in the input data, it is assumed this parameter is epsilon, and that
  omega = 1.395

  Inputs:
  datapath - Absolute path of data file containing a weak measurement tensor with indices
              [param, group, time, qubit, (mean, std, [true_params]), meas]
  data_group_size - Number of trajectories averaged together in the data file
  clean - If true, data in datapath is noise free
  stride - Subsample in time stride such that the input time index will be sliced according to [::stride]
  group_size - Number of trajectories averaged in each group of the output tensors
  num_train_groups - Only this many groups will be used for the training set, the rest will be discarded
  meas_op - The two measurement operator indices, e.g. [0,1] = [X,Y]
  debug - If true, more verbose output will be turned on

  Outputs:
  Training, validation, and test tensors with the following indices:
  [dataset]_x - [group, param, time, (qubit0, qubit1, meas_op0, meas_op1), meas]
  [dataset]_y - [group, param, time, (qubit0, qubit1, meas_op0, meas_op1), meas, (value, omega, eps)]
  [dataset]_params - [group, param, [true_params]]
  '''
  voltage = tf.saved_model.load(datapath)

  all_params = voltage[:,0,0,0,2:,0].numpy()
  voltage = voltage[...,0,:]
  
  # Append measurement operators
  if len(meas_op) == 0:
    # It is assumed that there are three measurements: XX, YY, and ZZ
    assert(voltage.shape[-1] == 3)
    voltage = tf.concat([voltage, 0.0*tf.ones_like(voltage)[...,:1,:], 0.0*tf.ones_like(voltage)[...,:1,:]], axis=3)
    voltage = voltage.numpy()
    voltage[:,:,:,2:,0] = 0.0
    voltage[:,:,:,2:,1] = 1.0
    voltage[:,:,:,2:,2] = 2.0
  else:
    assert(len(meas_op) == 2)
    voltage = tf.concat([voltage, meas_op[0]*tf.ones_like(voltage)[...,:1,:], meas_op[1]*tf.ones_like(voltage)[...,:1,:]], axis=3)
    voltage = voltage.numpy()

  # Subsample in time
  voltage = voltage[:,:,::stride,...]

  # Reshape to get voltage batches
  if clean:
    group_size = 1
    num_per_group = 1
    num_train_groups = 1
  else:
    num_per_group = int(group_size/data_group_size)
  all_x = voltage
  all_y = all_x

  # Split the voltages
  train_frac = 0.5
  train_x, valid_x, _, _ = fusion.split_data(all_x, all_y, train_frac)
  _, _, train_params, valid_params = fusion.split_data(all_x, all_params, train_frac)

  # Reduce the training to the requested number of groups and average the
  train_x = train_x[:,:num_train_groups*num_per_group,...]
  train_y = tf.repeat(tf.reduce_mean(train_x, axis=1)[:,tf.newaxis,...], num_train_groups*num_per_group, axis=1)

  # Split the validation data into valid and test
  num_valid_elms = valid_x.shape[1]
  if clean:
    test_x = valid_x
  else:
    test_x = valid_x[:,int(num_valid_elms/2):,...] # Test is back half
    valid_x = valid_x[:,:int(num_valid_elms/2),...] # Valid is first half

  # Validation set size should be half the training set size, unless they are both one
  if valid_x.shape[1] > num_train_groups*num_per_group/2:
    if num_train_groups >= 2:
      valid_x = valid_x[:,:int(num_train_groups*num_per_group/2),...]
    else:
      valid_x = valid_x[:,:num_train_groups*num_per_group,...]

  num_valid_groups = int(valid_x.shape[1]/num_per_group)
  num_test_groups = int(test_x.shape[1]/num_per_group)

  # Using x means as y data
  valid_y = tf.repeat(tf.reduce_mean(valid_x, axis=1)[:,tf.newaxis,...], num_valid_groups*num_per_group, axis=1)
  test_y = tf.repeat(tf.reduce_mean(test_x, axis=1)[:,tf.newaxis,...], num_test_groups*num_per_group, axis=1)

  train_x = tf.transpose(train_x, perm=[1,0,2,3,4])
  train_y = tf.transpose(train_y, perm=[1,0,2,3,4])
  valid_x = tf.transpose(valid_x, perm=[1,0,2,3,4])
  valid_y = tf.transpose(valid_y, perm=[1,0,2,3,4])
  test_x = tf.transpose(test_x, perm=[1,0,2,3,4])
  test_y = tf.transpose(test_y, perm=[1,0,2,3,4])
  train_params = tf.tile(train_params[tf.newaxis,:,:], multiples=[num_train_groups*num_per_group,1,1])
  test_params = tf.tile(valid_params[tf.newaxis,:,:], multiples=[num_test_groups*num_per_group,1,1])
  valid_params = tf.tile(valid_params[tf.newaxis,:,:], multiples=[num_valid_groups*num_per_group,1,1])

  # Train like denoising autoencoder solving for single Omega
  if train_params.shape[-1] == 1:
    # A single parameter is assumed to be epsilon with fixed Omega
    omega = 1.395
    train_y = tf.concat([train_y[...,tf.newaxis], tf.ones_like(train_y[...,tf.newaxis])*omega, tf.ones_like(train_y[...,tf.newaxis])*train_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,:]], axis=-1)
    valid_y = tf.concat([valid_y[...,tf.newaxis], tf.ones_like(valid_y[...,tf.newaxis])*omega, tf.ones_like(valid_y[...,tf.newaxis])*valid_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,:]], axis=-1)
    test_y = tf.concat([test_y[...,tf.newaxis], tf.ones_like(test_y[...,tf.newaxis])*omega, tf.ones_like(test_y[...,tf.newaxis])*test_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,:]], axis=-1)
  else:
    train_y = tf.concat([train_y[...,tf.newaxis], tf.ones_like(train_y[...,tf.newaxis])*train_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,:]], axis=-1)
    valid_y = tf.concat([valid_y[...,tf.newaxis], tf.ones_like(valid_y[...,tf.newaxis])*valid_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,:]], axis=-1)
    test_y = tf.concat([test_y[...,tf.newaxis], tf.ones_like(test_y[...,tf.newaxis])*test_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,:]], axis=-1)

  # Keep the real parts of the data only
  train_x = np.real(train_x)
  train_y = np.real(train_y)
  valid_x = np.real(valid_x)
  valid_y = np.real(valid_y)
  test_x = np.real(test_x)
  test_y = np.real(test_y)

  if debug:
    print('Data shapes:')
    print(train_x.shape)
    print(train_y.shape)
    print(train_params.shape)
    print(valid_x.shape)
    print(valid_y.shape)
    print(valid_params.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(test_params.shape)
  
  return train_x, train_y, train_params, valid_x, valid_y, valid_params, test_x, test_y, test_params

def train(train_x, train_y,                                            # Data
          valid_x, valid_y,
          test_x, test_y,
          init_ops, input_params, params, deltat, stride,              # Physical params
          group_size, data_group_size, groups_per_minibatch,           # Data size params
          start_run_idx, num_runs, num_epochs, num_eval_steps, lr, dr, # Train params
          historydir, modeldir,                                        # Output params
          perform_eval=True,
          savehist=True,
          savemodel=True,
          debug=True):
  '''
  Inputs:
  Data
  Training, validation, and test tensors with the following indices:
  [dataset]_x - [group, param, time, (qubit0, qubit1, meas_op0, meas_op1), meas]
  [dataset]_y - [group, param, time, (qubit0, qubit1, meas_op0, meas_op1), meas, (value, omega, eps)]

  Physical
  init_ops - The two measurement operator indices for the spin-up initial conditions, e.g. [0,1] = [X,Y]
  input_params - Indices into params array that are latent variables (encoder model output), e.g. [0,4] = omega, eps
  params - Physical parameter array [omega, gamma (2*kappa), eta, gamma_s, eps], 
           e.g. np.array([1.395,4.0*2.0*0.83156,0.1469,0.0,0.1], dtype=np.double)
  deltat - Time spacing in for the input/output measurement records
  stride - Subsample in time stride such that the input time index will be sliced according to [::stride]

  Data size
  group_size - Number of trajectories averaged in each group of the output tensors
  data_group_size - Number of trajectories averaged together in the data file
  groups_per_minibatch - Number of groups in each minibatch during training

  Train
  start_run_idx - Seed for the first run (determines output file labels as well)
  num_runs - Total number of runs
  num_epochs - List containing number of epochs to train for within each run, called a training run. Learning rate
               will be reset and metrics recorded at the end of each training run
  num_eval_steps - Number of random shuffles to perform when evaluating. Returned metrics will be averaged over
                   this number of steps
  lr - Learning rate during training
  dr - Decay rate during training, decay steps set to one

  Output
  historydir - Full path where histories will be saved
  modeldir - Full path where models will be saved
  perform_eval - Will perform random shuffle evaluation if true and add results to histories
  savehist - Flag indicating whether or not to save histories
  savemodel - Flag indicating whether or not to save models
  debug - Flag enabling verbose debug output
  '''
  num_per_group = int(group_size/data_group_size)
  phys_layer_idx = -6
  verbose_level = 1
  mini_batch_size = num_per_group*groups_per_minibatch

  # Setup model parameters
  _, params_per_group, seq_len, num_features, num_meas = train_x.shape
  num_features -= 2
  encoder_sizes = [100, 50]
  enc_lstm_size = 32
  dec_lstm_size = 16
  avg_size = max([1,int(20/stride)])
  num_traj = 1
  start_meas = 0.0
  comp_iq = True
  project_rho = False
  train_decoder = False
  strong_probs = []
  strong_probs_input = True
  num_params = len(input_params)

  max_val = 12
  offset = 1

  td_sizes = [32, 16]

  # Setup initial condition
  all_ops = sde_systems.paulis()
  rho0 = sde_systems.get_init_rho(all_ops[init_ops[0]], all_ops[init_ops[1]], 0, 0)
  if debug:
    pauli_names = ['X', 'Y', 'Z']
    print(f'Initial state: {pauli_names[init_ops[0]]}{pauli_names[init_ops[1]]}00')
    print('params:', params)

  num_training_runs = len(num_epochs)
  for run_idx in range(start_run_idx, start_run_idx + num_runs, 1):
    valid_metrics = []
    test_metrics = []

    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) `tensorflow` random seed
    # 3) `python` random seed
    seed = run_idx
    tf.keras.utils.set_random_seed(seed)

    # This will make TensorFlow ops as deterministic as possible, but it will
    # affect the overall performance, so it's not enabled by default.
    # `enable_op_determinism()` is introduced in TensorFlow 2.9.
    tf.config.experimental.enable_op_determinism()

    # Build RNN model
    model = flex.build_multimeas_rnn_model(seq_len, num_features, num_meas, avg_size, enc_lstm_size, dec_lstm_size, td_sizes, encoder_sizes, num_params,
                                           rho0, params, deltat, num_traj, start_meas, comp_iq=comp_iq, max_val=max_val, offset=offset,
                                           strong_probs=strong_probs, project_rho=project_rho, strong_probs_input=strong_probs_input,
                                           input_params=input_params, num_per_group=num_per_group, params_per_group=params_per_group)

    loss_func = fusion.fusion_mse_loss_shuffle

    metric_func = fusion.param_metric_shuffle_mse
    trimmed_metric_func = fusion.param_metric_shuffle_trimmed_mse
    omega_metric_func = fusion.param_metric_shuffle_omega_trimmed_mse
    eps_metric_func = fusion.param_metric_shuffle_eps_trimmed_mse

    all_metrics = [metric_func, trimmed_metric_func, omega_metric_func, eps_metric_func]
    fusion.compile_model(model, loss_func, metrics=all_metrics)

    for train_idx in range(num_training_runs):
      print(f'Training run {train_idx}')
      first_run = train_idx == 0
      if first_run:
        for layer in model.layers:
          layer.trainable = True
        model.layers[phys_layer_idx].trainable = train_decoder
        model.layers[phys_layer_idx].cell.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.a_cell_real.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.a_cell_imag.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.b_cell_real.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.b_cell_imag.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.a_dense_real.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.a_dense_imag.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.b_dense_real.trainable = train_decoder
        model.layers[phys_layer_idx].cell.flex.b_dense_imag.trainable = train_decoder

        fusion.compile_model(model, loss_func, metrics=all_metrics)

        model.summary()
        print(model.trainable_weights)

      lrscheduler = tf.keras.callbacks.LearningRateScheduler(tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=lr,
          decay_steps=1,
          decay_rate=dr))

      run_history = model.fit(train_x, train_y, batch_size=mini_batch_size, epochs=num_epochs[train_idx],
                              validation_data=(valid_x, valid_y), verbose=verbose_level, shuffle=True,
                              callbacks=[lrscheduler])

      if perform_eval:
        # Get the valid metric
        valid_vals = fusion.eval_model(model, valid_x, valid_y, num_eval_steps, num_per_group)
        vlosses = []
        vmetrics = []
        for d in valid_vals:
            vlosses += [d['loss']]
            vmetrics += [d['param_metric_shuffle_trimmed_mse']]
        valid_metrics += [valid_vals]
        print(f'Valid metric for run {train_idx}: {np.mean(vmetrics):.3g}')

        # Get the test metric
        test_vals = fusion.eval_model(model, test_x, test_y, num_eval_steps, num_per_group)
        tlosses = []
        tmetrics = []
        for d in test_vals:
            tlosses += [d['loss']]
            tmetrics += [d['param_metric_shuffle_trimmed_mse']]
        test_metrics += [test_vals]
        print(f'Test metric for run {train_idx}: {np.mean(tmetrics):.3g}')

      if first_run:
        history = run_history
      else:
        for k, v in run_history.history.items():
          history.history[k] += v

    # Save the history
    if savehist:
      history.history['seed'] = seed
      history.history['valid_metrics'] = valid_metrics
      history.history['test_metrics'] = test_metrics
      history.history['num_epochs'] = num_epochs
      savepath = historydir + f'hist_{seed}.dat'
      print('Saving history to', savepath)
      with open(savepath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Save the model
    if savemodel:
      savepath = os.path.join(modeldir, f'model_{seed}')
      print('Saving model to', savepath)
      fusion.save_model(model, savepath)