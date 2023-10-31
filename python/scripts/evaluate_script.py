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

currpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currpath,'../sdes'))
import sde_solve
import sde_systems
import data_gen

sys.path.append(os.path.join(currpath,'../systems'))
import rabi_weak_meas

sys.path.append(os.path.join(currpath,'../models'))
import fusion
import flex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Full path of the dataset')
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--group_size', required=True, type=int, help='Number of trajectories per group')
    parser.add_argument('--num_train_groups', required=True, type=int, help='Number of groups to use in training set')
    parser.add_argument('--groups_per_mb', required=False, default=1, type=int, help='Number of groups per minibatch')
    parser.add_argument('--seed', required=False, default=0, type=int, help='Random seed to use for the run')
    parser.add_argument('--stride', required=False, default=1, type=int, help='Time stride for cutting data file')
    parser.add_argument('--clean', action='store_true', help='If true, input data is clean, not sampled')

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
    if args.clean:
        print('Input data is noise free')
        omegas = tf.math.real(voltage[:,0,2,0])
        epsilons = tf.math.real(voltage[:,0,3,0])
        voltage = tf.math.real(voltage[:,tf.newaxis,...,:2,:])
    else:
        omegas = voltage[:,0,0,0,2,0]
        epsilons = voltage[:,0,0,0,3,0]
        voltage = voltage[...,0,:]
    all_params = tf.concat([omegas[:,tf.newaxis], epsilons[:,tf.newaxis]], axis=1).numpy()

    voltage = tf.concat([voltage, 0.0*tf.ones_like(voltage)[...,:1,:], 1.0*tf.ones_like(voltage)[...,:1,:]], axis=3)

    # Subsample in time
    stride = args.stride
    voltage = voltage[:,:,::stride,...]
    #all_probs = all_probs[:,::stride,:]

    mint = 0
    maxt = 4.0
    deltat = 2**(-8)*stride
    tvec = np.arange(mint,maxt,deltat)
    print('deltat:', deltat)

    # Reshape to get voltage batches
    if args.clean:
        group_size = 1
        num_per_group = 1
        num_train_groups = 1
    else:
        group_size = args.group_size
        num_per_group = int(group_size/100)
        num_train_groups = args.num_train_groups
    all_x = voltage
    all_y = all_x

    # Split the voltages
    train_frac = 0.5
    train_x, valid_x, _, _ = fusion.split_data(all_x.numpy(), all_y.numpy(), train_frac)
    _, _, train_params, valid_params = fusion.split_data(all_x.numpy(), all_params, train_frac)

    # Reduce the training to the requested number of groups and average the
    train_x = train_x[:,:num_train_groups*num_per_group,...]
    train_y = tf.repeat(tf.reduce_mean(train_x, axis=1)[:,tf.newaxis,...], num_train_groups*num_per_group, axis=1)

    # Split the validation data into valid and test
    num_valid_elms = valid_x.shape[1]
    if args.clean:
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

    # Set run parameters
    if args.clean:
        groups_per_minibatch = 1
        num_eval_steps = 1
    else:
        groups_per_minibatch = args.groups_per_mb
        num_eval_steps = 100
    phys_layer_idx = -6
    verbose_level = 1
    mini_batch_size = num_per_group*groups_per_minibatch
    num_epochs = [100, 100, 100]
    num_training_runs = len(num_epochs)
    lr = 3e-3
    dr = 0.99

    #_, seq_len, num_features, num_meas, num_strong_probs = train_x.shape
    _, params_per_group, seq_len, num_features, num_meas = train_x.shape
    num_features -= 2
    encoder_sizes = [100, 50]
    enc_lstm_size = 32
    dec_lstm_size = 16
    avg_size = max([1,int(20/args.stride)])
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

    seed = args.seed
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

    layer_name = 'param_layer'
    enc_model = tf.keras.Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
    enc_model.summary()

    modeldir = os.path.join(args.outdir,'models')
    historydir = os.path.join(args.outdir, 'histories')
    new_historydir = os.path.join(args.outdir, 'new_histories')
    if not os.path.exists(new_historydir):
        os.makedirs(new_historydir)

    fusion.load_model(enc_model, os.path.join(modeldir,f'model_{seed}'))

    loss_func = fusion.param_loss_omega_eps_shuffle
    omega_metric_func = fusion.param_loss_omega_trimmed_shuffle
    eps_metric_func = fusion.param_loss_eps_trimmed_shuffle

    all_metrics = [omega_metric_func, eps_metric_func]
    fusion.compile_model(enc_model, loss_func, metrics=all_metrics)

    valid_vals = fusion.eval_model(enc_model, valid_x, valid_params, num_eval_steps, num_per_group)
    test_vals = fusion.eval_model(enc_model, test_x, test_params, num_eval_steps, num_per_group)

    # Load the history
    history_file = f'hist_{seed}.dat'
    with open(os.path.join(historydir,history_file), "rb") as file_pi:
        history = pickle.load(file_pi)
    history['valid_metrics'] += [valid_vals]
    history['test_metrics'] += [test_vals]

    # Write the new history
    savepath = os.path.join(new_historydir, f'hist_{seed}.dat')
    print('Saving history to', savepath)
    with open(savepath, 'wb') as file_pi:
        pickle.dump(history, file_pi)

if __name__ == '__main__':
    main()