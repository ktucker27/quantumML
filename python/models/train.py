import numpy as np
import tensorflow as tf
import os
import sys

import fusion
import flex

def load_dataset(datapath, data_group_size, clean, stride, group_size, num_train_groups, meas_op=[], debug=True):
    '''
    Loads data from the specified path and splits it into training/validation/test sets

    Inputs:
    datapath - Absolute path of data file containing a weak measurement tensor with indices
               [param, group, time, qubit, (mean, std, [true_params]), meas]
    data_group_size - Number of trajectories averaged together in the data file
    Outputs:
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
