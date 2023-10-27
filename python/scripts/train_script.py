import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import os
import sys
import csv
import math
import time
import re
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed
from tensorflow.keras import backend as K
#from keras.layers.preprocessing import preprocessing_utils

sys.path.append('../sdes')
import sde_solve
import sde_systems
import data_gen

sys.path.append('../systems')
import rabi_weak_meas

sys.path.append('../models')
import fusion
import flex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Full path of the dataset')
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--seed', required=False, default=0, type=int, help='Random seed to use for the run')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load large dataset averaged over 10 runs
    mint = 0
    maxt = 4.0
    deltat = 2**(-8)
    tvec = np.arange(mint,maxt,deltat)
    voltage_dir = args.datapath

    voltage = tf.saved_model.load(voltage_dir)
    epsilons = np.arange(0.0, 2.0, 0.05)

    voltage = voltage[...,0,:]
    voltage = tf.concat([voltage, 0.0*tf.ones_like(voltage)[...,:1,:], 1.0*tf.ones_like(voltage)[...,:1,:]], axis=3)

    # Subsample in time
    stride = 1
    voltage = voltage[:,:,::stride,...]
    #all_probs = all_probs[:,::stride,:]

    mint = 0
    maxt = 4.0
    deltat = 2**(-8)*stride
    tvec = np.arange(mint,maxt,deltat)
    print('deltat:', deltat)

    # Reshape to get voltage batches
    group_size = 4000
    num_per_group = int(group_size/100)
    all_x = tf.reshape(voltage,[voltage.shape[0], -1, num_per_group, voltage.shape[2], voltage.shape[3], voltage.shape[4]])

    all_x_mean = tf.reduce_mean(tf.math.real(all_x), axis=2)
    all_y = all_x_mean

    # Split the voltages
    train_frac = 0.5
    train_x, valid_x, _, _ = fusion.split_data(all_x_mean.numpy(), all_y.numpy(), train_frac)
    _, _, train_params, valid_params = fusion.split_data(all_x_mean.numpy(), epsilons, train_frac)

    _, eval_valid_x, _, _ = fusion.split_data(voltage.numpy(), voltage.numpy(), train_frac)

    # Reduce the training to the requested number of groups and average the
    num_train_groups = 1
    train_x = train_x[:,:num_train_groups,...]
    train_y = tf.repeat(tf.reduce_mean(train_x, axis=1)[:,tf.newaxis,...], num_train_groups, axis=1)

    # Split the validation data into valid and test
    num_groups = valid_x.shape[1]
    test_x = eval_valid_x[:,int(num_groups*num_per_group/2):,...] # Test is back half
    eval_valid_x = eval_valid_x[:,:int(num_groups*num_per_group/2),...] # Valid is first half

    # Validation set size should be half the training set size, unless they are both one
    if valid_x.shape[1] > num_train_groups/2:
        if num_train_groups >= 2:
            valid_x = valid_x[:,:int(num_train_groups/2),...]
            eval_valid_x = eval_valid_x[:,:int(num_train_groups*num_per_group/2),...]
        else:
            valid_x = valid_x[:,:num_train_groups,...]
            eval_valid_x = eval_valid_x[:,:num_train_groups*num_per_group,...]

    num_valid_groups = valid_x.shape[1]
    num_test_groups = int(test_x.shape[1]/num_per_group)

    # Using x means as y data
    valid_y = tf.repeat(tf.reduce_mean(valid_x, axis=1)[:,tf.newaxis,...], num_valid_groups, axis=1)
    eval_valid_y = tf.repeat(tf.reduce_mean(eval_valid_x, axis=1)[:,tf.newaxis,...], num_valid_groups*num_per_group, axis=1)
    test_y = tf.repeat(tf.reduce_mean(test_x, axis=1)[:,tf.newaxis,...], num_test_groups*num_per_group, axis=1)

    train_x = tf.reshape(train_x, [-1, train_x.shape[2], train_x.shape[3], train_x.shape[4]])
    train_y = tf.reshape(train_y, [-1, train_y.shape[2], train_y.shape[3], train_y.shape[4]])
    valid_x = tf.reshape(valid_x, [-1, valid_x.shape[2], valid_x.shape[3], valid_x.shape[4]])
    valid_y = tf.reshape(valid_y, [-1, valid_y.shape[2], valid_y.shape[3], valid_y.shape[4]])
    eval_valid_x = tf.transpose(eval_valid_x, perm=[1,0,2,3,4])
    eval_valid_y = tf.transpose(eval_valid_y, perm=[1,0,2,3,4])
    test_x = tf.transpose(test_x, perm=[1,0,2,3,4])
    test_y = tf.transpose(test_y, perm=[1,0,2,3,4])
    train_params = tf.repeat(train_params, num_train_groups, axis=0)
    test_params = tf.tile(valid_params[tf.newaxis,:], multiples=[num_test_groups*num_per_group,1])
    valid_params = tf.repeat(valid_params, num_valid_groups, axis=0)
    eval_valid_params = tf.tile(valid_params[tf.newaxis,:], multiples=[num_valid_groups*num_per_group,1])

    # Train like denoising autoencoder solving for single Omega
    omega = 1.395
    train_y = tf.concat([train_y[...,tf.newaxis], tf.ones_like(train_y[...,tf.newaxis])*omega, tf.ones_like(train_y[...,tf.newaxis])*train_params[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis]], axis=-1)
    valid_y = tf.concat([valid_y[...,tf.newaxis], tf.ones_like(valid_y[...,tf.newaxis])*omega, tf.ones_like(valid_y[...,tf.newaxis])*valid_params[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis]], axis=-1)
    eval_valid_y = tf.concat([eval_valid_y[...,tf.newaxis], tf.ones_like(eval_valid_y[...,tf.newaxis])*omega, tf.ones_like(eval_valid_y[...,tf.newaxis])*eval_valid_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis]], axis=-1)
    test_y = tf.concat([test_y[...,tf.newaxis], tf.ones_like(test_y[...,tf.newaxis])*omega, tf.ones_like(test_y[...,tf.newaxis])*test_params[:,:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis]], axis=-1)

    # Keep the real parts of the data only
    train_x = np.real(train_x)
    train_y = np.real(train_y)
    valid_x = np.real(valid_x)
    valid_y = np.real(valid_y)
    eval_valid_x = np.real(eval_valid_x)
    eval_valid_y = np.real(eval_valid_y)
    test_x = np.real(test_x)
    test_y = np.real(test_y)

    print('Data shapes:')
    print(train_x.shape)
    print(train_y.shape)
    print(train_params.shape)
    print(valid_x.shape)
    print(valid_y.shape)
    print(valid_params.shape)
    print(eval_valid_x.shape)
    print(eval_valid_y.shape)
    print(eval_valid_params.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(test_params.shape)

    # Set run parameters
    num_runs = 1
    start_run_idx = args.seed

    verbose_level = 1
    mini_batch_size = 20
    num_epochs = [100, 100, 100]
    num_training_runs = len(num_epochs)
    num_eval_steps = 100
    lr = 3e-3
    dr = 0.99

    perform_eval = False
    savehist = True
    savemodel = True

    historydir = os.path.join(args.outdir,'4K_4K_dt2pm8/histories/')
    print(historydir, args.outdir) 
    if not os.path.exists(historydir):
        os.makedirs(historydir)
    
    modeldir = os.path.join(args.outdir,'4K_4K_dt2pm8/models/')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    #_, seq_len, num_features, num_meas, num_strong_probs = train_x.shape
    _, seq_len, num_features, num_meas = train_x.shape
    num_strong_probs = 0
    num_features -= 2
    conv_sizes = [32]
    encoder_sizes = [100, 50]
    enc_lstm_size = 32
    dec_lstm_size = 16
    avg_size = 20
    num_traj = 1
    start_meas = 0.0
    comp_iq = True
    project_rho = False
    train_decoder = False
    strong_probs = []
    strong_probs_input = True
    input_params = [0,4]
    num_params = 2

    max_val = 12
    offset = 1

    td_sizes = [32, 16]

    # Starting with ZZ00 initial condition
    sx, sy, sz = sde_systems.paulis()
    rho0 = sde_systems.get_init_rho(sx, sy, 0, 0)

    # Set the parameter values (with an omega error)
    params = np.array([1.395,4.0*2.0*0.83156,0.1469,0.0,0.1], dtype=np.double)

    valid_metrics = []
    test_metrics = []
    for run_idx in range(start_run_idx, start_run_idx + num_runs, 1):
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
                                            input_params=input_params)

      model.summary()

      print(model.trainable_weights)

      loss_func = fusion.fusion_mse_loss_voltage_xyz

      metric_func = fusion.param_metric_volt_xyz_mse
      omega_metric_func = fusion.param_metric_omega_mse
      eps_metric_func = fusion.param_metric_eps_mse

      trimmed_metric_func = fusion.param_metric_volt_xyz_trimmed_mse
      trimmed_omega_metric_func = fusion.param_metric_omega_trimmed_mse
      trimmed_eps_metric_func = fusion.param_metric_eps_trimmed_mse

      #trimmed_metric_func = fusion.param_metric_volt_xyz_trimmed_mse
      all_metrics = [metric_func, omega_metric_func, eps_metric_func]
      fusion.compile_model(model, loss_func, metrics=all_metrics)

      for train_idx in range(num_training_runs):
        print(f'Training run {train_idx}')
        first_run = train_idx == 0
        if first_run:
          for layer in model.layers:
            layer.trainable = True
          model.layers[-4].trainable = train_decoder
          model.layers[-4].cell.trainable = train_decoder
          model.layers[-4].cell.flex.a_cell_real.trainable = train_decoder
          model.layers[-4].cell.flex.a_cell_imag.trainable = train_decoder
          model.layers[-4].cell.flex.b_cell_real.trainable = train_decoder
          model.layers[-4].cell.flex.b_cell_imag.trainable = train_decoder
          model.layers[-4].cell.flex.a_dense_real.trainable = train_decoder
          model.layers[-4].cell.flex.a_dense_imag.trainable = train_decoder
          model.layers[-4].cell.flex.b_dense_real.trainable = train_decoder
          model.layers[-4].cell.flex.b_dense_imag.trainable = train_decoder

          fusion.compile_model(model, loss_func, metrics=all_metrics)

        lrscheduler = tf.keras.callbacks.LearningRateScheduler(tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=1,
            decay_rate=dr))

        run_history = model.fit(train_x, train_y, batch_size=mini_batch_size, epochs=num_epochs[train_idx],
                                validation_data=(valid_x, valid_y), verbose=verbose_level, shuffle=True,
                                callbacks=[lrscheduler])

        if perform_eval:
          # Get the valid metric
          valid_vals = fusion.eval_model(model, eval_valid_x, eval_valid_y, num_eval_steps, num_per_group)
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

if __name__ == '__main__':
    main()
