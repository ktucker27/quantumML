import numpy as np
import tensorflow as tf
import qutip as qt
import scipy
import os
import sys
import unittest
import csv
import time

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import fusion
import flex

pydir = os.path.dirname(parent)
sys.path.append(os.path.join(pydir, 'sdes'))

import sde_systems

repodir = os.path.dirname(pydir)
datadir = os.path.join(repodir, 'data')

# Set verbosity levels
test_tns_verbose = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def validate_density(rhovec, tol=1e-12):
    '''
    Input:
    rhovec - shape = [batch, pdim, pdim] density operators
    '''
    # Check trace
    success = True
    trace_err = tf.reduce_max(tf.abs(1.0 - tf.linalg.trace(rhovec)))
    print('Max trace abs error:', trace_err)
    if trace_err > tol:
        success = False
    
    # Check Hermiticity
    rho_dagger = tf.transpose(rhovec, perm=[0,2,1], conjugate=True)
    herm_err = tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(tf.abs(rhovec - rho_dagger)), axis=[1,2])))
    print('Max rho - rho* F-norm:', herm_err)
    if herm_err > tol:
        success = False
    
    # Check positivity
    evals, _ = tf.linalg.eig(rhovec)
    max_imag = tf.reduce_max(tf.abs(tf.math.imag(evals)))
    min_eval = tf.reduce_min(tf.math.real(evals))
    print('Max imag:', max_imag)
    print('Min eval:', min_eval)
    print('Evals checked:', evals.shape)
    if max_imag > tol or min_eval < -1.0*tol:
        success = False
    
    print('Pass:', success)
    return success

def evec_to_rho(lam, evec):
    rho = tf.zeros([4,4], tf.complex128)
    for ii in range(len(lam)):
        rho = rho + tf.cast(tf.complex(tf.math.real(lam[ii]), tf.math.imag(lam[ii])), tf.complex128)*tf.matmul(evec[:,ii,tf.newaxis], tf.transpose(evec[:,ii,tf.newaxis], conjugate=True))
    return rho

def rho_fnorm(lam, evec, mu):
  rho = evec_to_rho(lam, evec)
  return tf.reduce_sum(tf.square(tf.abs(rho - mu)))

def load_truth_file_two_qubits(filepath, return_dict):
    '''Loads a single tab delimited ground truth file'''
    
    print(f'Loading {filepath}')
    
    idx = filepath.find('Initial')
    prep_state = f"prep_{filepath[idx+7:idx+11].replace('=','')}"

    meas_idx = filepath.find('MeasurementOutcome')
    meas_outcome = f"meas_{filepath[meas_idx + len('MeasurementOutcome='):meas_idx + len('MeasurementOutcome=') + 1]}"

    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        rowidx = 0
        for row in reader:            
            if rowidx == 0:
                time_axis = np.array([x.split('_')[1] for x in row[1:] if len(x) > 0], dtype=float)
                if not 'time_axis' in return_dict.keys():
                    return_dict['time_axis'] = time_axis
                else:
                    tol = 1e-10
                    assert np.max(np.abs(return_dict['time_axis'] - time_axis)) < tol
            else:
                row_data = [x for x in row[1:] if len(x) > 0]
                if rowidx == 1:
                    true_probs = np.array(row_data, dtype=float)
                else:
                    true_probs = np.vstack((true_probs, np.array(row_data, dtype=float)))
                
            rowidx = rowidx + 1

    if not prep_state in return_dict.keys():
        return_dict[prep_state] = {}
    return_dict[prep_state][meas_outcome] = true_probs

def load_truth_data_two_qubits(filedir, return_dict, filepath_key=None):
    '''
    A function for loading ground truth probability data from tab-delimited files
    '''

    for filename in os.listdir(filedir):
        filepath = os.path.join(filedir, filename)
        if os.path.isfile(filepath):
            if filename.endswith('.dat') and (filepath_key is None or filepath.find(filepath_key) >= 0):
                load_truth_file_two_qubits(filepath, return_dict)
        #elif os.path.isdir(filepath):
        #    load_truth_data_two_qubits(filepath, return_dict)
            
def truth_dict_to_probs(truth_dict):
    '''
    Takes the ground truth dictionary and returns an np.array with shape
    [num_prep_states, num_time_steps, num_meas]
    '''
    meas_types = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    meas_outcomes = ['0', '1', '2', '3']
    #prep_states = [f'prep_{x}{y}' for x in meas_types for y in meas_outcomes]
    prep_states = ['prep_ZZ0']

    probs = np.zeros((len(truth_dict.keys()) - 1, truth_dict[prep_states[0]]['meas_0'].shape[1], 4*truth_dict[prep_states[0]]['meas_0'].shape[0]))

    for prep_idx in range(len(prep_states)):
        for meas_outcome in range(4):
            probs[prep_idx,:,[4*x + meas_outcome for x in range(9)]] = truth_dict[prep_states[prep_idx]][f'meas_{meas_outcome}']

    return truth_dict['time_axis'], probs

def init_probs_from_truth(truth_probs):
    '''
    Extracts initial probability distributions from the ground truth table
    and puts them in a dictionary used by the loss function
    '''
    meas_types = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    meas_outcomes = ['0', '1', '2', '3']
    prep_states = [f'prep_{x}{y}' for x in meas_types for y in meas_outcomes]

    return_dict = {}
    for prep_idx, prep_state in enumerate(prep_states):
        return_dict[prep_state] = {}
        for meas_idx, meas_type in enumerate(meas_types):
            return_dict[prep_state][f'meas_{meas_type}'] = np.reshape(truth_probs[prep_idx, 0, 4*meas_idx:4*meas_idx+4], 4)

    return return_dict

def diff_to_volt(diff, deltat):
  volt = np.zeros_like(diff.numpy())
  for tidx in range(diff.shape[-2]):
    if tidx < diff.shape[-2] - 1:
      volt[...,tidx+1,:] = volt[...,tidx,:] + deltat*diff[...,tidx,:]
    else:
      volt = np.concatenate([volt, volt[...,tidx:tidx+1,:] + deltat*diff[...,tidx:tidx+1,:]], axis=-2)
  return volt

class TestLiouv(unittest.TestCase):
    def test_liouv_vs_truth_files(self):
        tol = 1e-12

        # Load probabilities from truth files
        filedir = os.path.join(datadir, 'two_qubit_truth')
        truth_dict = {}
        load_truth_data_two_qubits(filedir, truth_dict)
        truth_times, truth_probs = truth_dict_to_probs(truth_dict)

        # Run Liouvillian evolution
        mint = 0.0
        maxt = 1.0
        deltat = truth_times[1]

        omega = 1.395
        kappa = 0.83156
        eta = 0.1469
        eps = 0.1

        _, _, sz = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sz, sz, 0, 0)
        liouv = sde_systems.RabiWeakMeasSDE.get_liouv(omega, 2.0*kappa, [eps], 2, [2,2])
        _, liouv_probs = sde_systems.get_2d_probs_truth(liouv, rho0, deltat, maxt - deltat*0.5)

        # Compare all probabilities
        mse = tf.reduce_mean(tf.pow(tf.math.real(truth_probs[0,:25,:] - liouv_probs[:,6:]), 2.0))
        print(f'Liouvillian vs. files MSE: {mse}')
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(liouv_probs))), 1e-14)
        self.assertLessEqual(mse, tol)

class TestRunModel2d(unittest.TestCase):

    def test_r2d_vs_truth_files(self):
        tol = 5e-4

        # Set initial condition
        _, _, sz = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sz, sz, 0, 0)[tf.newaxis,...]

        # Load probabilities from truth files
        filedir = os.path.join(datadir, 'two_qubit_truth')
        truth_dict = {}
        load_truth_data_two_qubits(filedir, truth_dict)
        truth_times, truth_probs = truth_dict_to_probs(truth_dict)

        # Run the model
        omega = 1.395
        kappa = 0.83156
        eta = 0.1469
        gamma_s = 0.0
        eps = 0.1
        num_traj=100

        input_params = [4]
        meas_op = [2,2]

        epsilons_traj = tf.repeat([eps], repeats=num_traj, axis=0)[:,tf.newaxis]
        params = np.array([omega,2.0*kappa,eta,gamma_s,eps], dtype=np.float32)
        for ii in range(params.shape[0]):
            if ii in input_params:
                param_idx = input_params.index(ii)
                param_inputs = epsilons_traj[:,param_idx:param_idx+1]
            else:
                param_inputs = params[ii]*np.ones_like(epsilons_traj[:,:1])

            if ii == 0:
                traj_inputs = param_inputs
            else:
                traj_inputs = tf.concat([traj_inputs, param_inputs], axis=1)

        meas_op0 = tf.one_hot([meas_op[0]], depth=3)*tf.ones([num_traj,3], tf.float32)
        meas_op1 = tf.one_hot([meas_op[1]], depth=3)*tf.ones([num_traj,3], tf.float32)
        traj_inputs = tf.concat([tf.cast(traj_inputs, tf.float32), meas_op0, meas_op1], axis=1)
        
        t0 = time.time()
        print('Running run_model_2d...')
        rhovec, _, _, tvec = fusion.run_model_2d(rho0, traj_inputs, num_traj, mint = 0.0, maxt = 1.0, deltat=2**(-8), comp_i=False)
        print(f'Done. Run time (s): {time.time() - t0}')
        probs = sde_systems.get_2d_probs(rhovec)

        # Compare all probabilities
        mses = np.zeros(36, dtype=np.double)
        for ii in range(36):
            mse = tf.abs(tf.reduce_mean(tf.pow(truth_probs[0,:25,ii] - np.interp(truth_times[:25], tvec, tf.reduce_mean(probs[:,:,ii+6], axis=0)), 2.0)))
            mses[ii] = mse
            self.assertLessEqual(mse, tol)
        print(f'Max MSE: {np.max(mses)}')

    def test_r2d_vs_liouv(self):
        tol = 5e-4

        mint = 0.0
        maxt = 1.0
        deltat = 2**(-8)

        omega = 1.395
        kappa = 0.83156
        eta = 0.1469
        gamma_s = 0.0
        epsilons = np.arange(0.0, 2.01, 0.5)
        num_traj=100

        input_params = [4]
        meas_op = [2,2]

        sx, _, _ = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sx, sx, 0, 1)[tf.newaxis,...]

        for eps in epsilons:
            epsilons_traj = tf.repeat([eps], repeats=num_traj, axis=0)[:,tf.newaxis]
            params = np.array([omega,2.0*kappa,eta,gamma_s,eps], dtype=np.float32)
            for ii in range(params.shape[0]):
                if ii in input_params:
                    param_idx = input_params.index(ii)
                    param_inputs = epsilons_traj[:,param_idx:param_idx+1]
                else:
                    param_inputs = params[ii]*np.ones_like(epsilons_traj[:,:1])

                if ii == 0:
                    traj_inputs = param_inputs
                else:
                    traj_inputs = tf.concat([traj_inputs, param_inputs], axis=1)

            meas_op0 = tf.one_hot([meas_op[0]], depth=3)*tf.ones([num_traj,3], tf.float32)
            meas_op1 = tf.one_hot([meas_op[1]], depth=3)*tf.ones([num_traj,3], tf.float32)
            traj_inputs = tf.concat([tf.cast(traj_inputs, tf.float32), meas_op0, meas_op1], axis=1)
            
            # Run the model
            print(f'epsilon = {eps}')
            t0 = time.time()
            print('Running run_model_2d...')
            rhovec, _, _, _ = fusion.run_model_2d(rho0, traj_inputs, num_traj, mint=mint, maxt=maxt, deltat=deltat, comp_i=False)
            print(f'Done. Run time (s): {time.time() - t0}')
            probs = sde_systems.get_2d_probs(rhovec)

            # Run the Liouvillian to get truth
            liouv = sde_systems.RabiWeakMeasSDE.get_liouv(omega, 2.0*kappa, [eps], 2)
            _, probs_truth = sde_systems.get_2d_probs_truth(liouv, rho0[0], deltat, maxt - deltat*0.5)

            # Compare all probabilities
            self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs))), 1e-16)
            self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs_truth))), 1e-14)
            mse = tf.reduce_mean(tf.pow(tf.math.real(tf.reduce_mean(probs, axis=0) - probs_truth), 2.0))
            print(f'MSE: {mse}')
            self.assertLessEqual(mse, tol)

    def test_r2d_vs_qutip(self):
        tol = 1e-2

        mint = 0.0
        maxt = 1.0
        deltat = 2**(-8)
        tvec = np.arange(mint,maxt,deltat)

        omega = 1.395
        kappa = 4.0*0.83156
        eta = 0.1469
        gamma_s = 0.0
        epsilons = np.arange(0.0, 2.01, 0.5)
        num_traj=1000
        meas_op = [0,1]
        input_params = [4]

        sx, sy, _ = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sx, sy, 0, 0)[tf.newaxis,...]

        for eps in epsilons:
            print(f'epsilon = {eps}')

            # Run QuTiP
            sx0 = qt.tensor(qt.sigmax(), qt.identity(2))
            sx1 = qt.tensor(qt.identity(2), qt.sigmax())
            sy0 = qt.tensor(qt.sigmay(), qt.identity(2))
            sy1 = qt.tensor(qt.identity(2), qt.sigmay())
            sz0 = qt.tensor(qt.sigmaz(), qt.identity(2))
            sz1 = qt.tensor(qt.identity(2), qt.sigmaz())
            szz = qt.tensor(qt.sigmaz(), qt.sigmaz())

            H = 0.5*omega*(sx0 + sx1) + eps*szz

            xup = (1.0/np.sqrt(2.0))*(qt.basis(2,0) + qt.basis(2,1))
            yup = (1.0/np.sqrt(2.0))*(qt.basis(2,0) + 1j*qt.basis(2,1))
            psi0 = qt.tensor(xup, yup)
            qtrho0 = psi0*psi0.dag()

            result = qt.smesolve(H, qtrho0, tvec,
                                c_ops=[np.sqrt(1.0 - 0.5*eta)*np.sqrt(kappa) * sx0, np.sqrt(1.0 - 0.5*eta)*np.sqrt(kappa) * sy1],
                                sc_ops=[np.sqrt(0.5*eta*kappa) * sx0, np.sqrt(0.5*eta*kappa) * sy1],
                                e_ops=[sx0,sx1,sy0,sy1,sz0,sz1],
                                ntraj=num_traj,
                                dW_factors=[1,1],
                                solver='euler',
                                store_measurement=True,
                                noise=1)

            qt_noise = np.array(result.noise)
            wvec = qt_noise[...,0,:,tf.newaxis]

            # Run the model
            epsilons_traj = tf.repeat([eps], repeats=num_traj, axis=0)[:,tf.newaxis]
            params = np.array([omega,2.0*kappa,eta,gamma_s,eps], dtype=np.float32)
            for ii in range(params.shape[0]):
                if ii in input_params:
                    param_idx = input_params.index(ii)
                    param_inputs = epsilons_traj[:,param_idx:param_idx+1]
                else:
                    param_inputs = params[ii]*np.ones_like(epsilons_traj[:,:1])

                if ii == 0:
                    traj_inputs = param_inputs
                else:
                    traj_inputs = tf.concat([traj_inputs, param_inputs], axis=1)

            meas_op0 = tf.one_hot([meas_op[0]], depth=3)*tf.ones([num_traj,3], tf.float32)
            meas_op1 = tf.one_hot([meas_op[1]], depth=3)*tf.ones([num_traj,3], tf.float32)
            traj_inputs = tf.concat([tf.cast(traj_inputs, tf.float32), meas_op0, meas_op1], axis=1)

            t0 = time.time()
            print('Running run_model_2d...')
            rhovec, ivec, _, _ = fusion.run_model_2d(rho0, traj_inputs, num_traj=num_traj, mint=mint, maxt=maxt, deltat=deltat, sim_noise=True, comp_i=True, wvec=wvec)
            print(f'Done. Run time (s): {time.time() - t0}')
            probs = sde_systems.get_2d_probs(rhovec)
            self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs))), 1e-16)
            probs = tf.math.real(probs)

            # Compare probabilities in the average
            mses = np.zeros(len(result.expect))
            for ii in range(len(result.expect)):
                mses[ii] = tf.reduce_mean(tf.square(tf.reduce_mean(probs[:,:,ii], axis=0) - 0.5*(result.expect[ii] + 1.0)))
            max_mse = np.max(mses)
            print(f'Max prob MSE (out of {len(result.expect)}):', max_mse)
            self.assertLessEqual(max_mse, 1.0e-14)

            # Compare trajectories relative to the mean
            all_meas = np.zeros([len(result.measurement), result.measurement[0].shape[0], result.measurement[0].shape[1]], result.measurement[0].dtype)
            for idx in range(len(result.measurement)):
                all_meas[idx,:,:] = result.measurement[idx]
            #qutip_diff = tf.reshape(all_meas, [-1,1,all_meas.shape[1], all_meas.shape[2]])
            #qutip_diff = tf.reduce_mean(qutip_diff, axis=1)
            
            qutip_volt = diff_to_volt(tf.constant(all_meas), deltat)[:,:-1,:]

            rel_err = tf.reduce_mean(tf.abs(ivec[:,20:,:] - qutip_volt[:,20:,:])/tf.abs(tf.reduce_mean(qutip_volt[:,20:,:], axis=0)))
            print(f'Rel ERR: {rel_err}')
            self.assertLessEqual(rel_err, tol)

class TestPhysicalRNN(unittest.TestCase):
    def test_rnn_layer(self):
        tol = 5e-4

        num_traj = 100
        mint = 0.0
        maxt = 1.0
        deltat = 2**(-8)
        tvec = np.arange(mint, maxt, deltat)

        omega = 1.395
        kappa = 0.83156
        eta = 0.1469
        gamma_s = 0.0
        epsilons = [0.0, 0.5, 1.0]
        params = np.array([omega,2.0*kappa,eta,gamma_s,epsilons[0]], dtype=np.double)
        meas_op = [2,2]

        sx, _, _ = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sx, sx, 0, 1)

        # Make the RNN layer
        lstm_size = 10
        a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
        a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
        b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
        b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

        a_rnn_cell_real.trainable = False
        a_rnn_cell_imag.trainable = False
        b_rnn_cell_real.trainable = False
        b_rnn_cell_imag.trainable = False

        repeat_layer = tf.keras.layers.RepeatVector(tvec.shape[0])
        #euler_cell = fusion.EulerRNNCell(rho0=tf.constant(rho0), maxt=1.5*deltat, deltat=deltat, params=params, num_traj=num_traj, input_param=4)
        euler_cell = flex.EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                           rho0=tf.constant(rho0), maxt=1.5*deltat, deltat=deltat, 
                                           params=params, num_traj=num_traj, input_param=[4], meas_param=1,
                                           project_rho=True, sim_noise=True)
        rnn_layer = tf.keras.layers.RNN(euler_cell,
                                        stateful=False,
                                        return_sequences=True,
                                        name='physical_layer')

        t0 = time.time()
        print('Running Euler RNN layer...')
        epsten = tf.constant(epsilons)
        meas_op0 = tf.one_hot([meas_op[0]], depth=3)*tf.ones([epsten.shape[0],3], tf.float32)
        meas_op1 = tf.one_hot([meas_op[1]], depth=3)*tf.ones([epsten.shape[0],3], tf.float32)
        traj_inputs = tf.concat([tf.cast(epsten[:,tf.newaxis], tf.float32), meas_op0, meas_op1], axis=1)
        repeat_out = repeat_layer(traj_inputs)
        probs = rnn_layer(repeat_out)
        print(f'Done. Run time (s): {time.time() - t0}')

        self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs))), 1e-16)

        # Run the Liouvillian
        for epsidx, eps in enumerate(epsilons):
            print(f'Checking epsilon = {eps}...')
            liouv = sde_systems.RabiWeakMeasSDE.get_liouv(omega, 2.0*kappa, [eps], 2, meas_op)
            _, probs_truth = sde_systems.get_2d_probs_truth(liouv, rho0, deltat, maxt + deltat*0.5)
            probs_truth = probs_truth[1:,:] # Exclude the initial point since the RNN output does

            # Compare all probabilities
            self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs_truth))), 1e-14)
            mse = tf.reduce_mean(tf.pow(probs[epsidx, :, :42] - tf.math.real(probs_truth), 2.0))
            print(f'MSE: {mse}')
            self.assertLessEqual(mse, tol)

    def test_rnn_layer_xyz(self):
        tol = 5e-6

        num_traj = 1
        mint = 0.0
        maxt = 1.0
        deltat = 2**(-8)
        tvec = np.arange(mint, maxt, deltat)

        omega = 1.395
        kappa = 0.83156
        eta = 0.1469
        gamma_s = 0.0
        epsilons = [0.0, 0.5, 1.0]
        num_eps = len(epsilons)
        params = np.array([0.0,2.0*kappa,eta,gamma_s,0.0], dtype=np.double)

        _, _, sz = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sz, sz, 0, 0)

        # Make the RNN layer
        lstm_size = 10
        a_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
        a_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
        b_rnn_cell_real = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')
        b_rnn_cell_imag = tf.keras.layers.LSTMCell(lstm_size, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')

        a_rnn_cell_real.trainable = False
        a_rnn_cell_imag.trainable = False
        b_rnn_cell_real.trainable = False
        b_rnn_cell_imag.trainable = False

        repeat_layer = tf.keras.layers.RepeatVector(tvec.shape[0])
        #euler_cell = fusion.EulerRNNCell(rho0=tf.constant(rho0), maxt=1.5*deltat, deltat=deltat, params=params, num_traj=num_traj, input_param=4)
        euler_cell = flex.EulerFlexRNNCell(a_rnn_cell_real, a_rnn_cell_imag, b_rnn_cell_real, b_rnn_cell_imag,
                                           rho0=tf.constant(rho0), maxt=1.5*deltat, deltat=deltat, 
                                           params=params, num_traj=num_traj, input_param=[0,4], meas_param=2,
                                           project_rho=True, sim_noise=False)
        rnn_layer = tf.keras.layers.RNN(euler_cell,
                                        stateful=False,
                                        return_sequences=True,
                                        name='physical_layer')

        # Setup input tensor
        epsten = tf.constant(epsilons)
        omegaten = omega*tf.ones([epsten.shape[0],1], tf.float32)
        paramten = tf.concat([omegaten, epsten[:,tf.newaxis]], axis=1)
        for meas_idx in range(3):
            meas_op = [meas_idx,meas_idx]
            meas_op0 = tf.one_hot([meas_op[0]], depth=3)*tf.ones([paramten.shape[0],3], tf.float32)
            meas_op1 = tf.one_hot([meas_op[1]], depth=3)*tf.ones([paramten.shape[0],3], tf.float32)
            meas_inputs = tf.concat([tf.cast(paramten, tf.float32), meas_op0, meas_op1], axis=1)
            if meas_idx == 0:
                traj_inputs = meas_inputs
            else:
                traj_inputs = tf.concat([traj_inputs, meas_inputs], axis=0)
        print('Input:')
        print(traj_inputs)
        
        # Run the layer
        t0 = time.time()
        print('Running Euler RNN layer...')
        repeat_out = repeat_layer(traj_inputs)
        probs = rnn_layer(repeat_out)
        print(f'Done. Run time (s): {time.time() - t0}')

        self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs))), 1e-16)

        # Run the Liouvillian
        for meas_idx in range(3):
            meas_op = [meas_idx,meas_idx]
            print(f'Checking measurement index {meas_idx}...')
            for epsidx, eps in enumerate(epsilons):
                print(f'Checking epsilon = {eps}...')
                liouv = sde_systems.RabiWeakMeasSDE.get_liouv(omega, 2.0*kappa, [eps], 2, meas_op)
                _, probs_truth = sde_systems.get_2d_probs_truth(liouv, rho0, deltat, maxt + deltat*0.5)
                probs_truth = probs_truth[1:,:] # Exclude the initial point since the RNN output does

                # Compare all probabilities
                self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(probs_truth))), 1e-14)
                mse = tf.reduce_mean(tf.pow(probs[meas_idx*num_eps + epsidx, :, :42] - tf.math.real(probs_truth), 2.0))
                print(f'MSE: {mse}')
                self.assertLessEqual(mse, tol)

class TestProjectToRho(unittest.TestCase):
    def test_project(self):
        # Create densities with negative eigenvalues and project

        tol = 1e-15
        scipy_tol = 1e-8

        tf.random.set_seed(10)

        n = 1000
        d = 4
        l = tf.cast(tf.complex(tf.random.uniform([n,d,d]), tf.random.uniform([n,d,d])), tf.complex128)
        mu = l + tf.transpose(l, perm=[0,2,1], conjugate=True)
        mu = mu/tf.linalg.trace(mu)[:,tf.newaxis,tf.newaxis]

        evals, evec = tf.linalg.eigh(mu)
        print(f'Matrices with negative evals: {tf.reduce_sum(tf.cast(tf.reduce_min(tf.math.real(evals), axis=1) < 0, tf.int32)).numpy()}/{n}')
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(evals))), tol)
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.reduce_sum(evals, axis=1) - 1)), 5e-6)
        evals = tf.math.real(evals)

        eval_abs_err = []
        rho_abs_err = []
        proj_rho = np.zeros_like(mu.numpy())
        for ii in range(n):
            if ii % 100 == 0:
                print(f'Processing {ii+1} out of {n}...')

            proj_rho[ii,:,:] = sde_systems.project_to_rho(mu[ii,:,:], d)
            rho_evals, _ = tf.linalg.eigh(proj_rho[ii,:,:])
            self.assertLessEqual(tf.reduce_max(tf.abs(tf.math.imag(rho_evals))), 1e-15)
            rho_evals = tf.math.real(rho_evals)

            # Get SciPy generated set of values satisfying constraints and minimizing two-norm with eigenvalues
            res = scipy.optimize.minimize(lambda x : np.sum((x - evals[ii,:].numpy())**2), evals[ii,:].numpy(), 
                                          constraints={'type':'eq', 'fun': lambda y : np.sum(y) - 1}, 
                                          bounds=[(0,'inf'), (0,'inf'), (0,'inf'), (0,'inf')], tol=1e-16)
            abs_err = np.max(np.abs(np.sort(res.x) - np.sort(rho_evals.numpy())))
            eval_abs_err += [abs_err]
            self.assertLessEqual(abs_err, scipy_tol)

            # Get SciPy generated nearest density to mu[ii,:,:] that minimizes (5) in the project_to_rho paper
            res = scipy.optimize.minimize(lambda x : rho_fnorm(x, evec[ii,:,:].numpy(), mu[ii,:,:]).numpy(), evals[ii,:].numpy(),
                                          constraints={'type':'eq', 'fun': lambda y : np.sum(y) - 1},
                                          bounds=[(0,'inf'), (0,'inf'), (0,'inf'), (0,'inf')], tol=1e-16)
            rho2 = evec_to_rho(res.x, evec[ii,:,:])
            abs_err2 = tf.reduce_max(tf.abs(rho2 - proj_rho[ii,:,:]))
            rho_abs_err += [abs_err2]
            if abs_err2 > scipy_tol:
                print('mu:', mu[ii,:,:])
                print('proj_rho:', proj_rho)
                print('rho2:', rho2)
            self.assertLessEqual(abs_err2, scipy_tol)

        # Validate density operators
        assert(validate_density(tf.constant(proj_rho)))

        # Validate SciPy comparison
        print('SciPy eval abs err (min, max, mean, num):', np.min(eval_abs_err), np.max(eval_abs_err), np.mean(eval_abs_err), len(eval_abs_err))
        print('SciPy eval abs err quantiles (0.0, .25, .50, .75, .95, 1.0):', np.quantile(eval_abs_err, [0.0, .25, .50, .75, .95, 1.0]))
        print('SciPy density abs err (min, max, mean, num):', np.min(rho_abs_err), np.max(rho_abs_err), np.mean(rho_abs_err), len(rho_abs_err))
        print('SciPy density abs err quantiles (0.0, .25, .50, .75, .95, 1.0):', np.quantile(rho_abs_err, [0.0, .25, .50, .75, .95, 1.0]))

    def test_valid_rho(self):
        # Run the projection on a set of valid density operators and confirm they do not change

        tol = 1e-14

        mint = 0.0
        maxt = 1.0
        deltat = 2**(-8)

        omega = 1.395
        kappa = 0.83156
        epsilons = [0.0, 0.5, 1.0]
        meas_op = [2,2]

        sx, _, _ = sde_systems.paulis()
        rho0 = sde_systems.get_init_rho(sx, sx, 0, 1)

        # Run the Liouvillian
        for epsidx, eps in enumerate(epsilons):
            print(f'Checking epsilon = {eps}...')
            liouv = sde_systems.RabiWeakMeasSDE.get_liouv(omega, 2.0*kappa, [eps], 2, meas_op)
            rhovec_truth, _ = sde_systems.get_2d_probs_truth(liouv, rho0, deltat, maxt - deltat*0.5)

            self.assertTrue(validate_density(tf.constant(rhovec_truth)))

            proj_rho = np.zeros_like(rhovec_truth)
            for ii in range(rhovec_truth.shape[0]):
                proj_rho[ii,:,:] = sde_systems.project_to_rho(tf.constant(rhovec_truth[ii,:,:]), 4).numpy()

            self.assertTrue(validate_density(tf.constant(proj_rho)))

            abs_err = tf.reduce_max(tf.abs(proj_rho - rhovec_truth))
            print(f'ABS ERR: {abs_err}')
            self.assertLessEqual(abs_err, tol)

if __name__ == '__main__':
    unittest.main(verbosity=2)