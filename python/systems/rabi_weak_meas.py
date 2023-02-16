import numpy as np
import tensorflow as tf
import csv
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

qubit_prep_dict = {"prep_X+" : {"prep_x" : [1.0, 0.0],
                                "prep_y" : [0.5, 0.5],
                                "prep_z" : [0.5, 0.5]},
                   "prep_X-" : {"prep_x" : [0.0, 1.0],
                                "prep_y" : [0.5, 0.5],
                                "prep_z" : [0.5, 0.5]},
                   "prep_Y+" : {"prep_x" : [0.5, 0.5],
                                "prep_y" : [1.0, 0.0],
                                "prep_z" : [0.5, 0.5]},
                   "prep_Y-" : {"prep_x" : [0.5, 0.5],
                                "prep_y" : [0.0, 1.0],
                                "prep_z" : [0.5, 0.5]},
                   "prep_Z+" : {"prep_x" : [0.5, 0.5],
                                "prep_y" : [0.5, 0.5],
                                "prep_z" : [1.0, 0.0]},
                   "prep_Z-": {"prep_x": [0.5, 0.5],
                               "prep_y": [0.5, 0.5],
                               "prep_z": [0.0, 1.0]},
                   "prep_g": {"prep_x": [0.5, 0.5],
                              "prep_y": [0.5, 0.5],
                              "prep_z": [1.0, 0.0]},
                   "prep_e": {"prep_x": [0.5, 0.5],
                              "prep_y": [0.5, 0.5],
                              "prep_z": [0.0, 1.0]}}

def load_tab_file(filepath, return_dict, durations, max_samples=np.inf, max_time=np.inf):
    '''Loads a single tab delimited data file updating return_dict'''
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        rowidx = 0
        for row in reader:            
            if rowidx == 0:
                init_row = row
            elif rowidx == 1:
                if float(init_row[-2][2:]) > max_time:
                    break
                    
                print(f'Loading {filepath}')

                #n = int(round(float(init_row[-2][2:]),0))
                idx = np.where(np.abs(durations - float(init_row[-2][2:])) < 1e-10)[0]
                n = idx[0] + 1 # Later code assumes a 1 based index on timesteps. TODO: Change to 0
                time_key = f't_{n}'

                meas_key = f'meas_{init_row[-1][0]}'
                if row[0] == '0':
                    prep_key = f'prep_{init_row[0][0]}+'
                else:
                    prep_key = f'prep_{init_row[0][0]}-'

                if prep_key not in return_dict.keys():
                    return_dict[prep_key] = {}

                if meas_key not in return_dict[prep_key].keys():
                    return_dict[prep_key][meas_key] = {}

                if time_key not in return_dict[prep_key][meas_key].keys():
                    return_dict[prep_key][meas_key][time_key] = {}

                return_dict[prep_key][meas_key][time_key]['time_axis'] = np.array([float(x) for x in [y[2:] for y in init_row[1:-1]]])
                return_dict[prep_key][meas_key][time_key]['dt'] = eval(init_row[2][2:])
                return_dict[prep_key][meas_key][time_key]['strong_meas'] = np.array([int(row[-1])])
                return_dict[prep_key][meas_key][time_key]['weak_meas'] = [[float(x) for x in row[1:-1]]]
            else:
                return_dict[prep_key][meas_key][time_key]['strong_meas'] = np.append(return_dict[prep_key][meas_key][time_key]['strong_meas'], int(row[-1]))
                return_dict[prep_key][meas_key][time_key]['weak_meas'] = np.append(return_dict[prep_key][meas_key][time_key]['weak_meas'], [[float(x) for x in row[1:-1]]], axis=0)
            rowidx = rowidx + 1
            
            if rowidx == max_samples + 1:
                break

def load_tab_data(filedir, durations, return_dict, max_samples=np.inf, max_time=np.inf, recursive=False):
    '''
    A function for loading measurement and voltage data from tab-delimited files
    (rather than h5)
    '''

    for filename in os.listdir(filedir):
        filepath = os.path.join(filedir, filename)
        if os.path.isfile(filepath):
            if filename.endswith('.dat'):
                load_tab_file(filepath, return_dict, durations, max_samples, max_time)
        elif recursive and os.path.isdir(filepath):
            load_tab_data(filepath, durations, return_dict, max_samples, max_time)

def get_raw_data(data_dict, timesteps, prep_state_encoding, meas_encoding):
    '''Takes data from a particular prep state and measurement dictionary 
    and returns the raw measurement data for flattening'''
    
    raw_weak_meas = list()
    reps_per_timestep = list()
    durations = list()
    
    num_prep_states = prep_state_encoding.shape[0]
    
    # Create the preparation/measurement binary encoding
    if num_prep_states == 1:
        one_hot = meas_encoding
    else:
        one_hot = np.zeros((meas_encoding.shape[0], num_prep_states + meas_encoding.shape[1]))
        one_hot[:, :num_prep_states] = np.tile(prep_state_encoding, (meas_encoding.shape[0], 1))
        one_hot[:, num_prep_states:] = meas_encoding
    
    # Extract measurements by timestep
    first = True
    for n in timesteps:
        time_dict = data_dict[f't_{n}']
        
        #print(f'Timestep: {n}')
        
        strong_meas = time_dict['strong_meas']
        reps_per_timestep.append(strong_meas.shape[0])
        durations.append(time_dict['time_axis'][-1])

        if first:
            labels = one_hot[strong_meas]
            first = False
        else:
            labels = np.vstack((labels, one_hot[strong_meas]))
            
        weak_meas = time_dict['weak_meas']
        for vals in weak_meas:
            raw_weak_meas.append(vals.tolist())
            
    return raw_weak_meas, labels, reps_per_timestep, durations
        
def flatten_data(data_dict, lmv=-1):
    '''Takes data in dictionary form and returns tensors suitable for the ML model. Output data
    tensors data_x and data_y have a grouping in the first index of the following order:
    (prep_state, meas, timestep, sample)
    The dimension of sample is not necessarily consistent for every combination of the prior
    indices, and this number is recorded in reps_per_timestep'''
    
    # Extract preparation, measurement, and timestep info from the data
    prep_states = np.sort(list(data_dict.keys()))
    num_prep_states = len(prep_states)
    
    meas_list = np.sort(list(data_dict[prep_states[0]].keys()))
    num_meas = len(meas_list)
    
    timesteps = np.sort([int(x[2:]) for x in data_dict[prep_states[0]][meas_list[0]].keys() if x[:2] == 't_'])
    
    reps_per_timestep = list()
    
    for prep_state_idx, prep_state in enumerate(prep_states):
        prep_state_data_dict = data_dict[prep_state]
        
        print(prep_state)
        
        # Create the prep state encoding
        prep_state_encoding = np.array([x == prep_state for x in prep_states])
        
        raw_weak_meas = list()
        for meas_idx, meas in enumerate(meas_list):
            meas_data_dict = prep_state_data_dict[meas]
            
            print(meas)

            # Create the meas encoding
            # UPGRADE: Modify this to work for general multi-qubit systems
            meas_encoding = np.full((2, 2*num_meas), lmv)
            meas_encoding[0, 2*meas_idx] = 1
            meas_encoding[0, 2*meas_idx + 1] = 0
            meas_encoding[1, 2*meas_idx] = 0
            meas_encoding[1, 2*meas_idx + 1] = 1

            # Get the raw data and append to the existing lists
            meas_raw_weak_meas, meas_labels, meas_rpts, durations = get_raw_data(meas_data_dict, timesteps, prep_state_encoding, meas_encoding)
            if meas_idx == 0:
                labels = meas_labels
            else:
                labels = np.vstack((labels, meas_labels))
                
            reps_per_timestep = reps_per_timestep + meas_rpts
            raw_weak_meas = raw_weak_meas + meas_raw_weak_meas
            
        # Pad the raw weak measurements and labels and add to data tensor
        sequence_lengths = np.array([len(x) for x in raw_weak_meas])
        # The maximum sequence length should be the same across prep states
        if prep_state_idx > 0:
            assert max_seq_length == np.max(sequence_lengths)
            
        max_seq_length = np.max(sequence_lengths)
        prep_state_enc_len = 1 + (prep_state_encoding.shape[0] if prep_state_encoding.shape[0] > 1 else 0)
        padded_x = tf.keras.preprocessing.sequence.pad_sequences(raw_weak_meas, padding='post', dtype='float32', value=lmv)
        prep_state_data_x = np.ndarray(shape=(padded_x.shape[0], max_seq_length, prep_state_enc_len), dtype=np.float32)
        prep_state_data_x[:, :, 0] = padded_x
        if prep_state_enc_len > 1:
            prep_state_data_x[:, :, 1:] = prep_state_encoding.astype(np.float32)
        
            # Mask the prep state encoding if there is no data
            for idx, seq_len in enumerate(sequence_lengths):
                prep_state_data_x[idx, seq_len:, 1:] = lmv
    
        # Unpack the labels into the data tensor at the right temporal index
        prep_state_data_y = np.full((labels.shape[0], max_seq_length, labels.shape[1]), lmv)
        for idx, seq_len in enumerate(sequence_lengths):
            prep_state_data_y[idx, seq_len - 1, :] = labels[idx, :].astype(np.int)
            
        if prep_state_idx == 0:
            data_x = prep_state_data_x
            data_y = prep_state_data_y
        else:
            data_x = np.vstack((data_x, prep_state_data_x))
            data_y = np.vstack((data_y, prep_state_data_y))
    
    return data_x, data_y, prep_states, meas_list, timesteps, durations, np.array(reps_per_timestep)

def split_data(data_x, data_y, train_frac):
    steps_per_val = int(1/(1 - train_frac))
    all_idcs = np.arange(data_x.shape[0])
    val_idcs = all_idcs[0::steps_per_val]
    train_idcs = np.delete(all_idcs, val_idcs)
    
    train_x = data_x[train_idcs, :, :]
    valid_x = data_x[val_idcs, :, :]
    
    train_y = data_y[train_idcs, :, :]
    valid_y = data_y[val_idcs, :, :]
    
    return train_x, valid_x, train_y, valid_y

def qubit_multi_prep_loss_function(y_true, y_pred, prep_states):
    init_x = np.array([qubit_prep_dict[key]['prep_x'] for key in prep_states])
    init_y = np.array([qubit_prep_dict[key]['prep_y'] for key in prep_states])
    init_z = np.array([qubit_prep_dict[key]['prep_z'] for key in prep_states])

    num_prep_states = prep_states.shape[0]
    mask_value = -1

    # Extract initial state information
    y_true_prep_encoding = y_true[..., :num_prep_states]
    y_true_ro_results = y_true[..., num_prep_states:]
    y_pred_prep_encoding = y_pred[..., :num_prep_states]
    y_pred_ro_results = y_pred[..., num_prep_states:]

    # Processing on the readout labels
    batch_size = K.cast(K.shape(y_true_ro_results)[0], K.floatx())
    # Finds out where a readout is available
    mask = K.cast(K.not_equal(y_true_ro_results, int(mask_value)), K.floatx())
    # First do a softmax (when from_logits = True) and then calculate the cross-entropy: CE_i = -log(prob_i)
    # where prob_i is the predicted probability for y_true_i = 1.0
    # Note: this assumes that each voltage record has exactly 1 label associated with it.
    pred_logits = K.reshape(tf.boolean_mask(y_pred_ro_results, mask), (batch_size, 2))
    true_probs = K.reshape(tf.boolean_mask(y_true_ro_results, mask), (batch_size, 2))
    CE = K.categorical_crossentropy(true_probs, pred_logits, from_logits=True)
    L_readout = K.sum(CE) / batch_size

    # Penalize deviation from the known initial state at the first time step
    # Do a softmax to get the predicted probabilities
    mask = K.cast(K.not_equal(y_true_prep_encoding, int(mask_value)), K.floatx())
    pred_encoding = K.reshape(tf.boolean_mask(y_pred_prep_encoding, mask), (batch_size, num_prep_states))
    true_encoding = K.reshape(tf.boolean_mask(y_true_prep_encoding, mask), (batch_size, num_prep_states))
    CE = K.categorical_crossentropy(true_encoding, pred_encoding, from_logits=True)
    #print(CE, batch_size)
    L_prep_encoding = K.sum(CE) / batch_size

    #print(true_encoding, tf.constant(init_x, dtype=K.floatx()))
    init_x2 = tf.linalg.matmul(tf.cast(true_encoding, dtype=tf.float32), tf.constant(init_x, dtype=K.floatx()))
    init_y2 = tf.linalg.matmul(tf.cast(true_encoding, dtype=tf.float32), tf.constant(init_y, dtype=K.floatx()))
    init_z2 = tf.linalg.matmul(tf.cast(true_encoding, dtype=tf.float32), tf.constant(init_z, dtype=K.floatx()))

    # I think this is useless, because this is enforced in the loss function above
    ## init_x_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, :2])
    ## init_y_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, 2:4])
    ## init_z_pred = K.softmax(y_pred_ro_results[:, self.data_points_for_prep_state, 4:])

    # This will enforce the x, y and z values of the prep state on the first sample.
    init_x_pred = K.softmax(y_pred_ro_results[:, 0, :2])
    init_y_pred = K.softmax(y_pred_ro_results[:, 0, 2:4])
    init_z_pred = K.softmax(y_pred_ro_results[:, 0, 4:])

    L_init_state = K.sqrt(K.square(init_x2 - init_x_pred)[0] + \
                          K.square(init_y2 - init_y_pred)[0] + \
                          K.square(init_z2 - init_z_pred)[0])

    # Constrain the purity of the qubit state < 1
    X_all_t = 1.0 - 2.0 * K.softmax(y_pred_ro_results[:, :, 0:2], axis=-1)[:, :, 1]
    Y_all_t = 1.0 - 2.0 * K.softmax(y_pred_ro_results[:, :, 2:4], axis=-1)[:, :, 1]
    Z_all_t = 1.0 - 2.0 * K.softmax(y_pred_ro_results[:, :, 4:6], axis=-1)[:, :, 1]
    L_outside_sphere = K.relu(K.sqrt(K.square(X_all_t) + K.square(Y_all_t) + K.square(Z_all_t)), threshold=1.0)

    # Force the state of average readout results to be equal to the strong readout results.
    lagrange_1 = tf.constant(1.0, dtype=K.floatx()) # Readout cross-entropy
    lagrange_2 = tf.constant(1.0, dtype=K.floatx()) # Initial state
    lagrange_3 = tf.constant(1.0, dtype=K.floatx()) # Purity constraint
    lagrange_4 = tf.constant(0.1, dtype=K.floatx()) # Prep state encoding

    return lagrange_1 * L_readout + lagrange_2 * L_init_state[0] + lagrange_3 * K.mean(L_outside_sphere) + lagrange_4 * L_prep_encoding

def masked_multi_prep_accuracy(y_true, y_pred, prep_states):
    n_levels = 2

    num_prep_states = prep_states.shape[0]
    mask_value = -1
    
    batch_size = K.shape(y_true)[0]
    # Finds out where a readout is available, mask has shape (batch_size, max_seq_length, 6) for qubits
    mask = K.not_equal(y_true[..., num_prep_states:], mask_value)
    # Selects logits with a readout, pred_logits has shape (batch_size, 2) for qubits
    pred_logits = K.reshape(tf.boolean_mask(y_pred[..., num_prep_states:], mask), (batch_size, n_levels))
    # Do a softmax to get the predicted probabilities, pred_probs has shape (batch_size, 2) for qubits
    pred_probs = K.softmax(pred_logits)
    # True readout results are [0, 1] or [1, 0] for qubits or [0, 0, 1], [1, 0, 0] or [0, 1, 0] for qutrits
    # Note: this assumes that each voltage record has exactly 1 label associated with it.
    true_probs = K.reshape(tf.boolean_mask(y_true[..., num_prep_states:], mask), (batch_size, n_levels))
    # Categorical accuracy returns a 1 when |pred_probs - true_probs| < 0.5 and else a 0.
    well_predicted = tf.keras.metrics.categorical_accuracy(true_probs, pred_probs)
    return tf.reduce_mean(well_predicted)

def learning_rate_schedule(epoch):
    epochs_per_annealing = 5
    reduce_learning_rate_after = 0
    init_learning_rate = 0.001
    learning_rate_epoch_constant = 7
    
    epoch = tf.math.floormod(epoch, epochs_per_annealing)
    if epoch < reduce_learning_rate_after:
        return init_learning_rate
    else:
        # Drops an order of magnitude every self.learning_rate_epoch_constant epochs
        return init_learning_rate * tf.math.exp((reduce_learning_rate_after - epoch) / learning_rate_epoch_constant)
    
def pairwise_softmax(y_pred, n_levels):
    """
    When training on labels from different tomography axes, each tomography axis has 2 results that sum to 1.0
    This function takes y_pred of the form [L0x, L1x, L0y, L1y, L0z, L1z], where Lix,y,z are the predicted logits, and
    converts the logits to probabilities in a pairwise fashion such that probabilities = [P0x, P1x, P0y, P1y, P0z, P1z]
    and P0i + P1i = 1 for i = x, y, z.
    Note: if n_levels = 3, we assume to only have Z measurements. In that case we can apply a regular softmax.
    :param y_pred: Predicted logits from the RNN. 3D array with shape (batch size, time steps, 6)
    :param n_levels: 2 for a qubit, 3 for a qutrit.
    :return: Array of probabilities
    """
    # In the case of qubits, we should do a pairwise softmax.
    if n_levels == 2:
        probabilities = np.zeros(np.shape(y_pred))
        batch_size, seq_length, _ = np.shape(y_pred)
        for k in [0, 1, 2]:  # px, py, pz for qubits
            numerator = np.exp(y_pred[:, :, 2 * k:2 * k + 2])
            denominator = np.expand_dims(np.sum(np.exp(y_pred[:, :, 2 * k:2 * k + 2]), axis=2), axis=2)
            probabilities[:, :, 2 * k:2 * k + 2] = numerator / denominator
    elif n_levels == 3:
        # For a qutrit we can use the standard Keras activation function, because for a single measurement axes,
        # the qutrit probabilities should add up to 1.0.
        probabilities = tf.keras.activations.softmax(K.constant(y_pred)).numpy()

    return probabilities

def build_model(seq_len, num_features, lmv, lstm_size, num_prep_states, num_meas):
    model = tf.keras.Sequential()
    
    # Add a masking layer to handle different weak measurement sequence lengths
    model.add(tf.keras.layers.Masking(mask_value=lmv, input_shape=(seq_len, num_features)))
    
    # Add the LSTM layer
    # TODO - Do we need regularization parameters?
    model.add(tf.keras.layers.LSTM(lstm_size,
                                   batch_input_shape=(seq_len, num_features),
                                   dropout=0.0,
                                   stateful=False,
                                   return_sequences=True))
    
    # Add a dense layer for the distribution as a time distributed layer
    # UPGRADE: Modify this layer to handle multiple qubits
    if num_prep_states == 1:
        prob_dist = tf.keras.layers.Dense(2*num_meas)
    else:
        prob_dist = tf.keras.layers.Dense(num_prep_states + 2*num_meas)
        
    model.add(tf.keras.layers.TimeDistributed(prob_dist))
    
    return model

def compile_model(model, prep_states, optimizer='adam'):
    model.compile(loss=lambda x,y : qubit_multi_prep_loss_function(x,y,prep_states), optimizer=optimizer,
                    metrics=[lambda x,y : masked_multi_prep_accuracy(x,y,prep_states)])

def prob_vs_time(valid_x, y_pred_probabilities, num_prep_states, lmv=-1):
    total_samps, num_times, num_meas = y_pred_probabilities.shape
    samps_per_prep_state = int(np.rint(total_samps/num_prep_states))
    
    tol = 1e-10
    assert abs(samps_per_prep_state - total_samps/num_prep_states) < tol
    
    y_pred_copy = np.copy(y_pred_probabilities)
    indcs = np.where(valid_x[:,:,0] == lmv)
    y_pred_copy[indcs] = 0
    mask = np.ones(shape=y_pred_copy.shape)
    mask[indcs] = 0
    
    start_idx = 0
    probs = np.zeros(shape=(num_prep_states, num_times, num_meas))
    for prep_idx in range(num_prep_states):
        probs[prep_idx, :, :] = np.sum(y_pred_copy[start_idx:(start_idx + samps_per_prep_state), :, :], axis=0)/np.sum(mask[start_idx:(start_idx + samps_per_prep_state), :, :], axis=0)
            
        start_idx = start_idx + samps_per_prep_state
    
    return probs

def true_prob_vs_time(y_pred_probabilities, num_prep_states):
    total_samps, num_times, num_meas = y_pred_probabilities.shape
    samps_per_prep_state = int(np.rint(total_samps/num_prep_states))
    
    lmv = -1
    
    tol = 1e-10
    assert abs(samps_per_prep_state - total_samps/num_prep_states) < tol
    
    y_pred_copy = np.copy(y_pred_probabilities)
    indcs = np.where(y_pred_probabilities == lmv)
    y_pred_copy[indcs] = 0
    mask = np.ones(shape=y_pred_copy.shape)
    mask[indcs] = 0
    
    start_idx = 0
    probs = np.zeros(shape=(num_prep_states, num_times, num_meas))
    for prep_idx in range(num_prep_states):
        num_vals = np.sum(mask[start_idx:(start_idx + samps_per_prep_state), :, :], axis=0)
        num_vals[np.where(num_vals == 0)] = 1
        probs[prep_idx, :, :] = np.sum(y_pred_copy[start_idx:(start_idx + samps_per_prep_state), :, :], axis=0)/num_vals
                    
        start_idx = start_idx + samps_per_prep_state
    
    return probs

def load_truth_file(filepath, return_dict):
    '''Loads a single tab delimited ground truth file'''
    
    print(f'Loading {filepath}')
    
    idx = filepath.find('Initial')
    prep_state = filepath[idx:idx+10]

    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        rowidx = 0
        for row in reader:            
            if rowidx == 0:
                time_axis = np.array([x.split('_')[1] for x in row[1:] if len(x) > 0], dtype=np.float)
                if not 'time_axis' in return_dict.keys():
                    return_dict['time_axis'] = time_axis
                else:
                    tol = 1e-10
                    assert np.max(np.abs(return_dict['time_axis'] - time_axis)) < tol
            else:
                row_data = [x for x in row[1:] if len(x) > 0]
                if rowidx == 1:
                    true_probs = np.array(row_data, dtype=np.float)
                else:
                    true_probs = np.vstack((true_probs, np.array(row_data, dtype=np.float)))
                
            rowidx = rowidx + 1

    return_dict[prep_state] = true_probs

def load_truth_data(filedir, return_dict, filepath_key=None):
    '''
    A function for loading ground truth probability data from tab-delimited files
    '''

    for filename in os.listdir(filedir):
        filepath = os.path.join(filedir, filename)
        if os.path.isfile(filepath):
            if filename.endswith('.dat') and (filepath_key is None or filepath.find(filepath_key) >= 0):
                load_truth_file(filepath, return_dict)
        #elif os.path.isdir(filepath):
        #    load_truth_data(filepath, return_dict)
            
def truth_dict_to_probs(truth_dict):
    '''
    Takes the ground truth dictionary and returns an np.array with shape
    [num_prep_states, num_time_steps, num_meas]
    '''

    prep_states = ['InitialX=0', 'InitialX=1', 'InitialY=0', 'InitialY=1', 'InitialZ=0', 'InitialZ=1']

    probs = np.zeros((len(truth_dict.keys()) - 1, truth_dict['InitialX=0'].shape[1], 2*truth_dict['InitialX=0'].shape[0]))

    for prep_idx in range(len(prep_states)):
        probs[prep_idx,:,[0,2,4]] = truth_dict[prep_states[prep_idx]]
        probs[prep_idx,:,[1,3,5]] = 1 - truth_dict[prep_states[prep_idx]]

    return truth_dict['time_axis'], probs
