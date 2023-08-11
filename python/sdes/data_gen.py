import numpy as np
import tensorflow as tf
import sde_systems
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'models'))

import fusion

def gen_noise_free(epsilons, mint, maxt, deltat, stride, start_meas=0):
    omega = 1.395
    kappa = 0.83156
    eta = 0.1469

    num_traj = epsilons.shape[0]
    params = np.array([omega,2.0*kappa,eta], dtype=np.float32)
    
    traj_inputs = tf.tile(params[tf.newaxis,:], multiples=[num_traj,1])
    traj_inputs = tf.concat([traj_inputs, epsilons[:,tf.newaxis]], axis=1)

    sx, sy, sz = sde_systems.paulis()
    rho0 = sde_systems.get_init_rho(sz, sz, 0, 0)[tf.newaxis,...]

    rhovec, ivec, _, _ = fusion.run_model_2d(rho0, traj_inputs, num_traj, mint=mint, maxt=maxt, deltat=deltat, sim_noise=False, start_meas=start_meas)
    probs = sde_systems.get_2d_probs(rhovec)
    probs = tf.math.real(probs)

    return ivec, probs

def gen_sde_data(epsilons, mint, maxt, deltat, stride, grp_size, batch_size=1, sim_noise=True, start_meas=0):
    '''
    Input:
    epsilons           - Epsilon values for simulations
    mint, maxt, deltat - Defines time grid
    stride             - Number of time points between strong measurements
    grp_size           - Number of trajectories per batch
    batch_size         - Number of batches per parameter combo
    sim_noise          - Whether or not to simulate noise in weak/strong measurements
    start_meas         - Time at which to turn on weak measurement

    Returns:
    voltages   - shape = [num_eps*batch_size, grp_size, num_times, num_qubits]
    all_probs  - shape = [num_eps*batch_size, num_times, 42 (6 first order, 36 second order probs)]
    traj_probs - shape = [num_eps*batch_size, grp_size, num_strong_meas_times, num_qubits]
    '''
    # Generate training data using randomly selected epsilon values
    omega = 1.395
    kappa = 0.83156
    eta = 0.1469

    max_tries = 10
    #tol = 1e-5
    tol = 1e-2
    imag_tol = 7e-2
    batch_size = 1

    sx, sy, sz = sde_systems.paulis()
    rho0 = sde_systems.get_init_rho(sz, sz, 0, 0)[tf.newaxis,...]

    voltage = None
    all_probs = None
    traj_probs = None
    for eidx, eps in enumerate(epsilons):
      print(f'eps = {eps}, {eidx+1}/{epsilons.shape[0]}')
      for batch_idx in range(batch_size):
        print(f'Batch = {batch_idx}')
        params = np.array([omega,2.0*kappa,eta,eps], dtype=np.float32)

        success = False
        num_tries = 0
        while not success and num_tries < max_tries:
          rhovec, ivec, _, _ = fusion.run_model_2d(rho0, params[tf.newaxis,:], grp_size, mint=mint, maxt=maxt, deltat=deltat, sim_noise=sim_noise, start_meas=start_meas)
          probs_traj = sde_systems.get_2d_probs(rhovec)
          probs = tf.math.reduce_mean(probs_traj, axis=0)
          probs_stride = tf.concat([probs_traj[:,::stride,...], probs_traj[:,-1,tf.newaxis,...]], axis=1)
          num_tries = num_tries + 1

          max_imag = tf.reduce_max(tf.abs(tf.math.imag(probs)))
          probs = tf.math.real(probs)
          probs_stride = tf.math.real(probs_stride)
          max_prob = tf.reduce_max(probs)
          min_prob = tf.reduce_min(probs)
          if np.isnan(max_prob) or max_prob > 1.0 + tol or min_prob < -1.0*tol or max_imag > imag_tol:
            print(f'WARNING: Failed to generate valid probabilities for eps = {eps}, try {num_tries}/{max_tries}')
            print(f'Max prob: {max_prob}, Min prob: {min_prob}, Max imag: {max_imag}')
            continue

          success = True
          if voltage is None:
            voltage = ivec[tf.newaxis,...]
            all_probs = probs[tf.newaxis,...]
            traj_probs = probs_stride[tf.newaxis,...]
          else:
            voltage = tf.concat((voltage, ivec[tf.newaxis,...]), axis=0)
            all_probs = tf.concat((all_probs, probs[tf.newaxis,...]), axis=0)
            traj_probs = tf.concat((traj_probs, probs_stride[tf.newaxis,...]), axis=0)

        if not success:
          print(f'Failed for eps={eps}')
          break
        else:
          print('voltage shape:', voltage.shape)
          print('all_probs shape:', all_probs.shape)
          print('traj_probs shape:', traj_probs.shape)

      if not success:
        break

    if not success:
      voltage = None
      all_probs = None
      traj_probs = None

    print(f'Success = {success}')

    return voltage, all_probs, traj_probs