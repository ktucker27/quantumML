import numpy as np
import tensorflow as tf
import os
import sys
import pickle
import argparse

import fusion
import flex

currpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currpath,'../sdes'))

import sde_systems

MAX_TRAIN_RUNS = 100

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
    if debug:
      print('Assuming XX, YY, ZZ measurements')
  else:
    assert(len(meas_op) == 2)
    voltage = tf.concat([voltage, meas_op[0]*tf.ones_like(voltage)[...,:1,:], meas_op[1]*tf.ones_like(voltage)[...,:1,:]], axis=3)
    voltage = voltage.numpy()
    if debug:
      print('Meas ops:', meas_op)

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

def setup_model(seed, group_size, data_group_size,                 # Data size params
                params_per_group, seq_len, num_features, num_meas,
                init_ops, input_params, params, deltat, stride,    # Physical params
                encoder_only=False, debug=True, all_metrics=None,
                train_decoder=False):
  '''
  Creates a new model initialized with the given random seed

  Inputs:

  Data size
  seed - Random seed to set before model initialization
  params_per_group - Dimension of data parameter index
  seq_len - Dimension of data time index
  num_features - (Number of qubits)*(Measurement dimension per qubit) (typically just 2*1 = 2)
  num_meas - Dimension of data measurement index

  Physical
  init_ops - The two measurement operator indices for the spin-up initial conditions, e.g. [0,1] = [X,Y]
  input_params - Indices into params array that are latent variables (encoder model output), e.g. [0,4] = omega, eps
  params - Physical parameter array [omega, gamma (2*kappa), eta, gamma_s, eps], 
           e.g. np.array([1.395,4.0*2.0*0.83156,0.1469,0.0,0.1], dtype=np.double)
  deltat - Time spacing for the input/output measurement records
  stride - Subsample in time stride such that the input time index will be sliced according to [::stride]

  encoder_only - If true, the model will include only the encoder up to the physical parameter layer
  debug - Flag enabling verbose debug output
  all_metrics - List of metric functions to use during training and evaluation. Defaults to trimmed metrics
                ignoring first and last 10% of parameter estimates
  train_decoder - Flag indicating whether decoder variables are trainable

  Outputs:
  model - The initialized and compiled model, ready to train
  '''
  # Setup model parameters
  num_per_group = int(group_size/data_group_size)
  phys_layer_idx = -6

  encoder_sizes = [100, 50]
  enc_lstm_size = 32
  dec_lstm_size = 16
  avg_size = max([1,int(20/stride)])
  num_traj = 1
  start_meas = 0.0
  comp_iq = True
  project_rho = train_decoder
  strong_probs = []
  strong_probs_input = False
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

  # Set the seed using keras.utils.set_random_seed. This will set:
  # 1) `numpy` seed
  # 2) `tensorflow` random seed
  # 3) `python` random seed
  tf.keras.utils.set_random_seed(seed)

  # This will make TensorFlow ops as deterministic as possible, but it will
  # affect the overall performance, so it's not enabled by default.
  # `enable_op_determinism()` is introduced in TensorFlow 2.9.
  tf.config.experimental.enable_op_determinism()

  # Build RNN model
  model = flex.build_multimeas_rnn_model(seq_len, num_features, num_meas, avg_size, enc_lstm_size, dec_lstm_size, td_sizes, encoder_sizes, num_params,
                                          rho0, params, deltat, num_traj, start_meas, comp_iq=comp_iq, max_val=max_val, offset=offset,
                                          strong_probs=strong_probs, project_rho=project_rho, strong_probs_input=strong_probs_input,
                                          input_params=input_params, num_per_group=num_per_group, params_per_group=params_per_group,
                                          encoder_only=encoder_only)

  if encoder_only:
    loss_func = fusion.encoder_only_loss_shuffle

    if all_metrics is None:
      omega_metric_func = fusion.encoder_only_loss_omega_trimmed_shuffle
      eps_metric_func = fusion.encoder_only_loss_eps_trimmed_shuffle

      if len(input_params) > 1:
        all_metrics = [omega_metric_func, eps_metric_func]
      else:
        all_metrics = [eps_metric_func]
  else:
    # Setup loss and metric functions
    loss_func = fusion.fusion_mse_loss_shuffle

    if all_metrics is None:
      metric_func = fusion.param_metric_shuffle_mse
      trimmed_metric_func = fusion.param_metric_shuffle_trimmed_mse
      omega_metric_func = fusion.param_metric_shuffle_omega_trimmed_mse
      eps_metric_func = fusion.param_metric_shuffle_eps_trimmed_mse

      if len(input_params) > 1:
        all_metrics = [metric_func, trimmed_metric_func, omega_metric_func, eps_metric_func]
      else:
        all_metrics = [metric_func, trimmed_metric_func, eps_metric_func]

    # Set decoder trainability
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

  if debug:
    model.summary()
    print(model.trainable_weights)
  
  return model

def train_model(model, seed,
                train_x, train_y,                                     # Data
                valid_x, valid_y,
                test_x, test_y,
                group_size, data_group_size, groups_per_minibatch,    # Data size params
                num_epochs, stop_loss_thresh, num_eval_steps, lr, dr, # Train params
                perform_eval=True,
                debug=True):
  '''
  Trains the given model, evaluates it, and records results

  Inputs:
  model - The Keras model to train
  seed - Random seed used for model initialization (will determine output file names)

  Data
  Training, validation, and test tensors with the following indices:
  [dataset]_x - [group, param, time, (qubit0, qubit1, meas_op0, meas_op1), meas]
  [dataset]_y - [group, param, time, (qubit0, qubit1, meas_op0, meas_op1), meas, (value, omega, eps)]

  Data size
  group_size - Number of trajectories averaged in each group of the output tensors
  data_group_size - Number of trajectories averaged together in the data file
  groups_per_minibatch - Number of groups in each minibatch during training

  Train
  num_epochs - List containing number of epochs to train in each training run. Learning rate
               will be reset and metrics recorded at the end of each training run
  stop_loss_thresh - If > 0, training will continue until the final training run drop in validation loss and run-to-run
                     drop are greater than this threshold (see fusion.analyze_loss_conv for metrics). Minimum training
                     runs is 2 and maximum is 20
  num_eval_steps - Number of random shuffles to perform when evaluating. Returned metrics will be averaged over
                   this number of steps
  lr - Learning rate during training
  dr - Decay rate during training, decay steps set to one

  Output
  perform_eval - Will perform random shuffle evaluation if true and add results to histories
  debug - Flag enabling verbose debug output

  Outputs:
  history - History object containing lists of all losses and metrics concatenated across training runs 
            as well as the following fields:
              + seed - Records the seed
              + num_epochs - Records the number of epochs list
              + valid_metrics - List of num_eval_steps dictionaries with evaluation results for validation set
                                at the end of each training run (shape = [num_training_runs][num_eval_steps])
              + test_metrics - List of num_eval_steps dictionaries with evaluation results for test set
                               at the end of each training run (shape = [num_training_runs][num_eval_steps])
  '''
  num_per_group = int(group_size/data_group_size)
  verbose_level = 1
  mini_batch_size = num_per_group*groups_per_minibatch

  if stop_loss_thresh > 0:
    assert(not isinstance(num_epochs, list))
    num_training_runs = MAX_TRAIN_RUNS
  else:
    assert(isinstance(num_epochs, list))
    num_training_runs = len(num_epochs)

  valid_metrics = []
  test_metrics = []
  for train_idx in range(num_training_runs):
    if isinstance(num_epochs, list):
      train_run_epochs = num_epochs[train_idx]
    else:
      train_run_epochs = num_epochs

    if debug:
      print(f'Training run {train_idx}')
    first_run = train_idx == 0

    lrscheduler = tf.keras.callbacks.LearningRateScheduler(tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=1,
        decay_rate=dr))

    run_history = model.fit(train_x, train_y, batch_size=mini_batch_size, epochs=train_run_epochs,
                            validation_data=(valid_x, valid_y), verbose=verbose_level, shuffle=True,
                            callbacks=[lrscheduler])

    if perform_eval:
      # Get the valid metric
      valid_vals = fusion.eval_model(model, valid_x, valid_y, num_eval_steps, num_per_group)
      valid_metrics += [valid_vals]
      if debug:
        vmetrics = []
        for d in valid_vals:
            # Debug metrics selected to align with legacy scripts
            if 'param_metric_shuffle_trimmed_mse' in d.keys():
              vmetrics += [d['param_metric_shuffle_trimmed_mse']]
            elif 'param_loss_eps_trimmed_shuffle' in d.keys():
              vmetrics += [d['param_loss_eps_trimmed_shuffle']]
        if len(vmetrics) > 0:
          print(f'Valid metric for run {train_idx}: {np.mean(vmetrics):.3g}')

      # Get the test metric
      test_vals = fusion.eval_model(model, test_x, test_y, num_eval_steps, num_per_group)
      test_metrics += [test_vals]
      if debug:
        tmetrics = []
        for d in test_vals:
            # Debug metrics selected to align with legacy scripts
            if 'param_metric_shuffle_trimmed_mse' in d.keys():
              tmetrics += [d['param_metric_shuffle_trimmed_mse']]
            elif 'param_loss_eps_trimmed_shuffle' in d.keys():
              tmetrics += [d['param_loss_eps_trimmed_shuffle']]
        if len(tmetrics) > 0:
          print(f'Test metric for run {train_idx}: {np.mean(tmetrics):.3g}')

    if first_run:
      history = run_history
    else:
      for k, v in run_history.history.items():
        history.history[k] += v

    # Update evaluation history
    history.history['valid_metrics'] = valid_metrics
    history.history['test_metrics'] = test_metrics

    # Stop training if validation loss has saturated
    if not first_run and stop_loss_thresh > 0:
      train_run_ratio, last_run_ratio = fusion.analyze_loss_conv(history.history)
      print(f'Run {train_idx} stop ratios:', train_run_ratio, last_run_ratio)
      if train_run_ratio >= stop_loss_thresh and last_run_ratio >= stop_loss_thresh:
        break

  # Add fields to the history
  history.history['seed'] = seed
  history.history['num_epochs'] = num_epochs
  
  return history

def train(datapath, clean, num_train_groups,                           # Data params
          group_size, data_group_size, groups_per_minibatch,
          init_ops, meas_op, input_params, params, deltat, stride,     # Physical params
          start_run_idx, num_runs, num_epochs, stop_loss_thresh,       # Train params
          num_eval_steps, lr, dr,
          historydir, modeldir,                                        # Output params
          encoder_only=False, perform_eval=True,
          debug=True, all_metrics=None, train_decoder=False):
  '''
  Performs full training procedure: loading data, initializing the model, training the model, and
  recording results

  Inputs:
  Data
  datapath - Absolute path of data file containing a weak measurement tensor with indices
              [param, group, time, qubit, (mean, std, [true_params]), meas]
  clean - If true, data in datapath is noise free
  num_train_groups - Only this many groups will be used for the training set, the rest will be discarded
  group_size - Number of trajectories averaged in each group of the output tensors
  data_group_size - Number of trajectories averaged together in the data file
  groups_per_minibatch - Number of groups in each minibatch during training

  Physical
  init_ops - The two measurement operator indices for the spin-up initial conditions, e.g. [0,1] = [X,Y]
  meas_op - The two measurement operator indices, e.g. [0,1] = [X,Y]
  input_params - Indices into params array that are latent variables (encoder model output), e.g. [0,4] = omega, eps
  params - Physical parameter array [omega, gamma (2*kappa), eta, gamma_s, eps], 
           e.g. np.array([1.395,4.0*2.0*0.83156,0.1469,0.0,0.1], dtype=np.double)
  deltat - Time spacing for the measurement records in the data file
  stride - Subsample in time stride such that the input time index will be sliced according to [::stride]

  Train
  start_run_idx - Seed for the first run (determines output file labels as well)
  num_runs - Total number of runs
  num_epochs - List containing number of epochs to train for within each run, called a training run. Learning rate
               will be reset and metrics recorded at the end of each training run (single number if stop_loss_thresh > 0)
  stop_loss_thresh - If > 0, training will continue until the final training run drop in validation loss and run-to-run
                     drop are greater than this threshold (see fusion.analyze_loss_conv for metrics). Minimum training
                     runs is 2 and maximum is 20
  num_eval_steps - Number of random shuffles to perform when evaluating. Returned metrics will be averaged over
                   this number of steps
  lr - Learning rate during training
  dr - Decay rate during training, decay steps set to one

  Output
  historydir - Full path where histories will be saved
  modeldir - Full path where models will be saved
  perform_eval - Will perform random shuffle evaluation if true and add results to histories
  encoder_only - If true, the model will include only the encoder up to the physical parameter layer
  debug - Flag enabling verbose debug output
  all_metrics - List of metric functions to use during training and evaluation. Defaults to use number
                of parameters in dataset to select 1D or 2D trimmed metrics
  train_decoder - Flag indicating whether decoder variables are trainable
  '''
  # Override training parameters for clean data
  if clean:
    group_size = 1
    data_group_size = 1
    num_train_groups = 1
    groups_per_minibatch = 1
    num_eval_steps = 1

  # Load the data
  train_x, train_y, train_params, \
  valid_x, valid_y, valid_params, \
  test_x, test_y, test_params = load_dataset(datapath, data_group_size, clean, stride,
                                             group_size, num_train_groups, meas_op, debug)

  if encoder_only:
    train_y = train_params
    valid_y = valid_params
    test_y = test_params
    if train_params.shape[-1] == 1 and len(input_params) > 1:
      # Need to append fixed Omega in this case
      omega = 1.395
      train_y = tf.concat([tf.ones_like(train_y)*omega, train_y], axis=-1)
      valid_y = tf.concat([tf.ones_like(valid_y)*omega, valid_y], axis=-1)
      test_y = tf.concat([tf.ones_like(test_y)*omega, test_y], axis=-1)
    if debug:
      print('Using param tensors for Y values since model is encoder only')

  # Data loaders are including Omega in Y tensors. If we are only solving for epsilon
  # we need to drop the Omega values
  if input_params == [4]:
    assert(tf.shape(train_y)[-1] > 2)
    train_y = tf.gather(train_y, [0,2], axis=-1)
    valid_y = tf.gather(valid_y, [0,2], axis=-1)
    test_y = tf.gather(test_y, [0,2], axis=-1)

    if debug:
      print('Dropping Omega from Y tensors. New shapes:')
      print(train_y.shape)
      print(valid_y.shape)
      print(test_y.shape)

  # Setup metric functions if none were provided
  if all_metrics is None:
    if train_params.shape[-1] == 1 or len(input_params) == 1:
      # Solving for one parameter, use 1D metrics
      if encoder_only:
        omega_metric_func = fusion.encoder_only_loss_omega_trimmed_shuffle
        eps_metric_func = fusion.encoder_only_loss_eps_trimmed_shuffle
        if len(input_params) > 1:
          all_metrics = [omega_metric_func, eps_metric_func]
        else:
          all_metrics = [eps_metric_func]
      else:
        metric_func = fusion.param_metric_shuffle_mse
        trimmed_metric_func = fusion.param_metric_shuffle_trimmed_mse
        omega_metric_func = fusion.param_metric_shuffle_omega_trimmed_mse
        eps_metric_func = fusion.param_metric_shuffle_eps_trimmed_mse
        if len(input_params) > 1:
          all_metrics = [metric_func, trimmed_metric_func, omega_metric_func, eps_metric_func]
        else:
          all_metrics = [metric_func, trimmed_metric_func, eps_metric_func]
    else:
      # Solving for two parameters, use 2D metrics
      if encoder_only:
        omega_metric_func = fusion.encoder_only_loss_omega_2d_trimmed_shuffle
        eps_metric_func = fusion.encoder_only_loss_eps_2d_trimmed_shuffle
        all_metrics = [omega_metric_func, eps_metric_func]
      else:
        metric_func = fusion.param_metric_shuffle_mse
        trimmed_metric_func = fusion.param_metric_shuffle_2d_trimmed_mse
        omega_metric_func = fusion.param_metric_shuffle_omega_2d_trimmed_mse
        eps_metric_func = fusion.param_metric_shuffle_eps_2d_trimmed_mse
        all_metrics = [metric_func, trimmed_metric_func, omega_metric_func, eps_metric_func]

  _, params_per_group, seq_len, num_features, num_meas = train_x.shape
  num_features -= 2

  # Adjust the deltat from the file time spacing to the model time spacing according
  # to the given stride
  deltat = deltat*stride
  if debug:
    print('deltat:', deltat)

  # Loop over requested seeds
  for run_idx in range(start_run_idx, start_run_idx + num_runs, 1):
    # Setup model
    seed = run_idx
    model = setup_model(seed, group_size, data_group_size,
                        params_per_group, seq_len, num_features, num_meas,
                        init_ops, input_params, params, deltat, stride,
                        encoder_only, debug, all_metrics, train_decoder)
    
    # Train the model
    history = train_model(model, seed,
                          train_x, train_y, valid_x, valid_y, test_x, test_y,
                          group_size, data_group_size, groups_per_minibatch,
                          num_epochs, stop_loss_thresh, num_eval_steps, lr, dr,
                          perform_eval, debug)
    
    # Save the history
    if historydir is not None:
      if not os.path.exists(historydir):
        os.makedirs(historydir)

      savepath = os.path.join(historydir, f'hist_{seed}.dat')
      if debug:
        print('Saving history to', savepath)
      with open(savepath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Save the model
    if modeldir is not None:
      if not os.path.exists(modeldir):
        os.makedirs(modeldir)

      savepath = os.path.join(modeldir, f'model_{seed}')
      if debug:
        print('Saving model to', savepath)
      fusion.save_model(model, savepath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Full path of the dataset')
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--group_size', required=True, type=int, help='Number of trajectories per group')
    parser.add_argument('--data_group_size', required=True, type=int, help='Number of trajectories averaged together in the data file')
    parser.add_argument('--num_train_groups', required=True, type=int, help='Number of groups to use in training set')
    parser.add_argument('--groups_per_mb', required=False, default=1, type=int, help='Number of groups per minibatch')
    parser.add_argument('--num_epochs', required=True, type=str, help='A list of epochs per training run, or a single number if a stopping criteria is to be used')
    parser.add_argument('--seed', required=False, default=0, type=int, help='Random seed to use for the run')
    parser.add_argument('--deltat', required=True, type=str, help='Time interval for data file')
    parser.add_argument('--kappa', required=False, default=4.0*0.83156, type=float, help='Decoder model value of kappa')
    parser.add_argument('--gamma_s', required=False, default=0.0, type=float, help='Decoder model value of gamma_s')
    parser.add_argument('--stride', required=False, default=1, type=int, help='Time stride for cutting data file')
    parser.add_argument('--clean', action='store_true', help='If true, input data is clean, not sampled')
    parser.add_argument('--encoder_only', action='store_true', help='True in the supervised case where only the encoder is used')
    parser.add_argument('--use_1d_metrics', action='store_true', help='If true, metrics will be 1D regardless of number of parameters in the dataset')
    parser.add_argument('--train_decoder', action='store_true', help='Flag indicating whether variables in the decoder are trainable')
    parser.add_argument('--init_ops', required=False, default='[0,1]', type=str, help='A list specifying the initial spins that are up with [0,1,2]=[X,Y,Z]')
    parser.add_argument('--meas_op', required=False, default='[0,1]', type=str, help='A list specifying the measurement operators with [0,1,2]=[X,Y,Z]. [] implies three measurements: XX, YY, and ZZ')
    parser.add_argument('--input_params', required=False, default='[0,4]', type=str, help='A list specifying the parameters to solve for with [0,1,2,3,4]=[omega, kappa, eta, gamma_s, eps]')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Train params
    num_eval_steps = 100
    lr = 3e-3
    dr = 0.99
    perform_eval = True
    num_runs = 1

    num_epochs = eval(args.num_epochs)
    stop_loss_thresh = 0.95
    if isinstance(num_epochs, list):
      print('Epochs:', num_epochs)
      stop_loss_thresh = 0
    else:
      print('Epochs:', num_epochs, 'with stopping threshold', stop_loss_thresh)

    # XY initial conditions with XY measurements
    if args.init_ops is None:
      init_ops = [0,1]
    else:
      init_ops = eval(args.init_ops)
      if not isinstance(init_ops, list) or len(init_ops) != 2:
        print('init_ops must be a list of length two')
        sys.exit(1)
    
    if args.meas_op is None:
      meas_op = [0,1]
    else:
      meas_op = eval(args.meas_op)
      if not isinstance(init_ops, list):
        print('meas_op must be a list')
        sys.exit(1)

    # Params
    if args.input_params is None:
      input_params = [0,4]
    else:
      input_params = eval(args.input_params)
      if not isinstance(input_params, list):
        print('input_params must be a list')
        sys.exit(1)
    params = np.array([1.395,2.0*args.kappa,0.1469,args.gamma_s,0.1], dtype=np.double)
    deltat = eval(args.deltat)

    # Output
    historydir = os.path.join(args.outdir,'histories/')
    modeldir = os.path.join(args.outdir,'models/')

    # Metrics
    all_metrics = None
    if args.use_1d_metrics:
      if args.encoder_only:
        omega_metric_func = fusion.encoder_only_loss_omega_trimmed_shuffle
        eps_metric_func = fusion.encoder_only_loss_eps_trimmed_shuffle
        if len(input_params) > 1:
          all_metrics = [omega_metric_func, eps_metric_func]
        else:
          all_metrics = [eps_metric_func]
      else:
        metric_func = fusion.param_metric_shuffle_mse
        trimmed_metric_func = fusion.param_metric_shuffle_trimmed_mse
        omega_metric_func = fusion.param_metric_shuffle_omega_trimmed_mse
        eps_metric_func = fusion.param_metric_shuffle_eps_trimmed_mse
        if len(input_params) > 1:
          all_metrics = [metric_func, trimmed_metric_func, omega_metric_func, eps_metric_func]
        else:
          all_metrics = [metric_func, trimmed_metric_func, eps_metric_func]

    train(args.datapath, args.clean, args.num_train_groups,                  # Data params
          args.group_size, args.data_group_size, args.groups_per_mb,
          init_ops, meas_op, input_params, params, deltat, args.stride, # Physical params
          args.seed, num_runs, num_epochs, stop_loss_thresh,                 # Train params
          num_eval_steps, lr, dr,
          historydir=historydir, modeldir=modeldir,                          # Output params
          encoder_only=args.encoder_only, perform_eval=perform_eval,
          debug=True, all_metrics=all_metrics, train_decoder=args.train_decoder)

if __name__ == '__main__':
    main()