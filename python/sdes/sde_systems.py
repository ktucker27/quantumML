import math
import os
import sys
import numpy as np
import tensorflow as tf
import scipy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'tns'))

import operations

def paulis():
    return [ np.array([[0.0, 1.0],[1.0, 0.0]], dtype=np.cdouble), np.array([[0.0, -1.0j],[1.0j, 0.0]], dtype=np.cdouble), np.array([[1.0, 0.0],[0.0, -1.0]], dtype=np.cdouble) ]

def kron(a,b):
    #assert tf.rank(a) == 2
    #assert tf.rank(b) == 2
    cp = tf.tensordot(a,b,axes=0)
    c = tf.transpose(cp, perm=[0,2,1,3])
    return tf.reshape(c, [a.shape[0]*b.shape[0], a.shape[1]*b.shape[1]])

def calc_op_exp(rho,o):
    '''
    Input:
    rho: shape = [num_traj,num_times,pdim,pdim] set of density operators
    o: shape = [pdim,pdim] Hermetian operator to take expectation values of

    Returns:
    exp_o: shape = [num_traj,num_times] operator expectations
    '''

    return tf.linalg.trace(tf.matmul(rho, o))

def calc_exp(x,o):
    '''
    Input:
    x: shape = [num_traj,num_times,4] set of density operators
    o: shape = [2,2] Hermetian operator to take expectation values of

    Returns:
    exp_o: shape = [num_traj,num_times] operator expectations
    '''

    # Unwrap rho
    rho = tf.reshape(x, [-1,tf.shape(x)[1],2,2])

    return tf.linalg.trace(tf.matmul(rho, o))

def get_init_rho(op1, op2, idx1, idx2):
    '''
    Well return the pure state density operator corresponding to the requested tensor product
    of operator eigenvectors. E.g. providing (sz, sz, 0, 1) will select the (+-) eigenvectors
    for sz on the A and B subsystems respectively, put them in a tensor Kronecker product, and 
    return the outer product of that result

    Input:
    op1, op2 - shape = [sqrt(pdim), sqrt(pdim)] operators on A and B subsystems
    idx1, idx2 - indices of each eigenvalue to select taken from the list sorted
                 in descending order

    Output:
    rho0 - shape = [pdim, pdim] density operator for the pure state evec1[:,idx1] x evec2[:,idx2]
           where the eigenvectors are sorted in descending order of eigenvalues
    '''
    # Get eigenvalues/vectors
    evals1, evecs1 = np.linalg.eig(op1)
    eidx1 = np.flip(np.argsort(evals1))
    evals2, evecs2 = np.linalg.eig(op2)
    eidx2 = np.flip(np.argsort(evals2))

    # Get the vector in the tensor product space
    evec = tf.reshape(tf.tensordot(evecs1[:,eidx1[idx1]],evecs2[:,eidx2[idx2]],0), [-1])

    # Get the outer product of the eigenvector
    rho0 = tf.tensordot(evec, tf.math.conj(evec), 0)

    return rho0

def calc_op_probs(rho, op1, op2):
  '''
  Takes a batch of density operators and returns probabilities of each combination
  of measurement outcomes of the two passed in operators for subsystems A and B
  such that rho \in B(H), H = A x B, dim(A) = dim(B) = sqrt(pdim), and
  pdim = dim(H)

  Input:
  rho - shape = [num_traj, num_times, pdim, pdim] batched density operators
  op1, op2 - shape = [sqrt(pdim), sqrt(pdim)] operators on A and B subsystems

  Output:
  vals - shape = [pdim] eigenvalue of op1 x op2 where
         vals[i*pdim + j]= lambda1_i*lambda2_j and the lambdak_i are in
         descending order. E.g. for a 2 qubit system this order is
         (++), (+-), (-+), (--)
  probs - shape = [num_traj, num_times, pdim] probabilities corresponding to
          the above values
  '''

  # Get eigenvalues/vectors
  evals1, evecs1 = np.linalg.eig(op1)
  eidx1 = np.flip(np.argsort(evals1))
  evals2, evecs2 = np.linalg.eig(op2)
  eidx2 = np.flip(np.argsort(evals2))

  # Calculate probabilities
  vals = None
  probs = None
  for idx1 in eidx1:
    for idx2 in eidx2:
      # Get the eigenvector of the tensor product of operators with a Kronecker product
      evec = tf.reshape(tf.tensordot(evecs1[:,idx1],evecs2[:,idx2],0), [-1])

      # Get the projection operator onto the eigenvector and calculate its
      # expectation to get the probability
      op = tf.tensordot(evec, tf.math.conj(evec), 0)
      op_probs = calc_op_exp(rho, op)[:,:,tf.newaxis]

      if vals is None:
        vals = evals1[idx1]*evals2[idx2]
        probs = op_probs
      else:
        vals = np.append(vals, evals1[idx1]*evals2[idx2])
        probs = tf.concat([probs, op_probs], axis=2)

  return vals, probs

def get_2d_probs(rhovec):
  sx, sy, sz = paulis()

  ops = [sx, sy, sz]
  eye = np.eye(2, dtype=np.cdouble)

  probs = None
  for op in ops:
    prod_op1 = kron(op, eye)
    probvec = 0.5*(calc_op_exp(rhovec, prod_op1)[:,:,tf.newaxis] + 1.0)

    if probs is None:
        probs = probvec
    else:
        probs = tf.concat([probs, probvec], axis=2)

    prod_op2 = kron(eye, op)
    probvec = 0.5*(calc_op_exp(rhovec, prod_op2)[:,:,tf.newaxis] + 1.0)
    probs = tf.concat([probs, probvec], axis=2)

  for op1 in ops:
    for op2 in ops:
      _, probvec = calc_op_probs(rhovec, op1, op2)
      probs = tf.concat([probs, probvec], axis=2)

  return probs

def get_2d_probs_truth(liouv, rho0, deltat, maxt):
    rho0vec = np.reshape(rho0, [-1])

    dtmat = scipy.linalg.expm(liouv*deltat)

    rhovec_truth = rho0[:,:,np.newaxis]
    rhotvec = rho0vec
    for t in np.arange(deltat, maxt + 0.5*deltat, deltat):
        rhotvec = np.matmul(dtmat, rhotvec)
        rhot = np.reshape(rhotvec, [4,4])[:,:,np.newaxis]
        rhovec_truth = np.concatenate([rhovec_truth, rhot], axis=2)
    rhovec_truth = np.transpose(rhovec_truth, axes=[2,0,1])

    probs_truth = get_2d_probs(rhovec_truth[np.newaxis,...])[0,...]

    return rhovec_truth, probs_truth

def project_to_rho(mu, d):
    '''
    Projects Hermetian, trace-one but not necessarily non-negative mu to the nearest valid physical state (i.e.
    non-negative definite) rho

    Taken from algorithm for subproblem 1 in:
    Efficient Method for Computing the Maximum-Likelihood Quantum State from Measurements with Additive Gaussian Noise
    John A. Smolin, Jay M. Gambetta, and Graeme Smith
    Phys. Rev. Lett. 108, 070502 - Published 17 February 2012

    Inputs:
    mu - shape = [d, d]

    Outputs:
    rho - Same shape as mu, but non-negative definite
    '''

    # Get the eigenvalues/vectors of mu
    evals, evec = tf.linalg.eig(mu)
    #assert(tf.reduce_max(tf.abs(tf.math.imag(evals))) < 1.0e-10)
    evals = tf.cast(tf.math.real(evals), tf.float64)
    evalsidx = tf.argsort(evals, direction='DESCENDING', axis=-1)

    a = tf.zeros(1, dtype=tf.float64)
    lam = tf.zeros(d, dtype=tf.float64)
    ii = d-1
    for eidx in tf.reverse(evalsidx, axis=[0]):
        if tf.gather(evals, eidx) + a/tf.cast(ii+1, dtype=tf.float64) > 0.0:
            break
        a += tf.gather(evals, eidx)
        ii = ii - 1

    for jj in range(ii+1):
        lam = lam + tf.one_hot(jj, d, dtype=tf.float64)*(evals[evalsidx[jj]] + a/tf.cast(ii+1, tf.float64))
    
    rho = tf.zeros([d,d], dtype=tf.complex128)
    for ii in range(d):
        rho = rho + tf.cast(lam[ii], tf.complex128)*tf.matmul(evec[:,evalsidx[ii],tf.newaxis], tf.transpose(evec[:,evalsidx[ii],tf.newaxis], conjugate=True))

    return rho

def unwrap_x_to_rho(x, pdim):
    '''
    Takes a tensor storing the upper triangle of rho in row major order and reshapes it
    into density operators
    Inputs:
    x - shape = [num_traj, pdim(pdim+1)/2]
    Outputs:
    rho - shape = [num_traj, pdim, pdim] where elements below the diagonal are the complex
          conjugates of their counterparts in x
    '''
    num_traj = tf.shape(x)[0]
    #pdim = int(-0.5 + tf.math.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
    permidx = [int(pdim*ii + jj - ii*(ii+1)/2) if ii <= jj else int(pdim*(pdim+1)/2) for ii in range(pdim) for jj in range(pdim)]
    x2 = tf.concat([x,tf.zeros([num_traj,1], dtype=x.dtype)], axis=1)
    x3 = tf.gather(x2, permidx, axis=1)
    x4 = tf.reshape(x3, [-1,pdim,pdim])
    rho = (x4 + tf.transpose(x4, perm=[0,2,1], conjugate=True))*(tf.ones([num_traj,pdim,pdim], dtype=x4.dtype) - 0.5*tf.linalg.eye(pdim, dtype=x4.dtype))
    return rho

def wrap_rho_to_x(rho, pdim):
    '''
    Takes the upper triangle of rho and reshapes it to a vector in row major order
    Inputs:
    rho - shape = [num_traj, pdim, pdim]
    Outputs:
    x - shape = [num_traj, pdim*(pdim+1)/2]
    '''
    #pdim = int(tf.shape(rho)[1])
    vecsize = int(pdim*(pdim+1)/2)
    rhovec = tf.reshape(rho, [-1, pdim**2])
    permidx = tf.concat([[int(pdim*ii + jj) for ii in range(pdim) for jj in range(ii,pdim)], tf.zeros(int(pdim**2 - vecsize), dtype=tf.int32)], axis=0)
    x = tf.gather(rhovec, permidx, axis=1)[:,:vecsize]
    return x

class GeometricSDE:
    def a(t,x,p):
        return p[0]*x

    def b(t,x,p):
        return p[1]*x

    def bp(t,x,p):
        return p[1]*tf.ones([tf.shape(x)[0],1,1,1])

class Geometric2DSDE:
    def a(t,x,p):
        return tf.reshape((tf.constant([[1.0],[0.0]])*p[0] + tf.constant([[0.0],[1.0]])*p[1]), [1,2,1])*x

    def b(t,x,p):
        return tf.reshape(tf.constant([[1.0, 0.0],[0.0, 0.0]])*p[2] + tf.constant([[0.0, 0.0],[0.0, 1.0]])*p[3],[1,2,2])*x

    def bp(t,x,p):
        num_traj = tf.shape(x)[0]
        return tf.tile(tf.expand_dims(tf.stack([tf.constant([[1.0, 0.0],[0.0, 0.0]])*p[2],tf.constant([[0.0, 0.0],[0.0, 1.0]])*p[3]]), axis=0), [num_traj,1,1,1])

def supd_herm(l,rho):
        '''
        NB: It is assumed that l is Hermetian as conjugate transposes are omitted
        '''
        d1 = tf.matmul(l,tf.matmul(rho,l))
        l2 = tf.matmul(l,l)
        d2 = 0.5*(tf.matmul(l2,rho) + tf.matmul(rho,l2))
        return d1 - d2

def suph_herm(l,rho):
    '''
    NB: It is assumed that l is Hermetian as conjugate transposes are omitted
    '''
    h1 = tf.matmul(l,rho) + tf.matmul(rho,l)
    h2 = tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])*rho
    return h1 - h2

def suph_herm_p(l,rho):
    '''
    NB: It is assumed that l is Hermetian as conjugate transposes are omitted
    '''
    # return shape = [num_traj,d,d]
    pdim = tf.shape(l)[0]
    t1 = kron(l, tf.eye(pdim, dtype=rho.dtype))
    t2 = kron(tf.eye(pdim, dtype=rho.dtype), l)

    rho_vec = tf.reshape(rho, [-1,pdim**2])
    lvec = tf.reshape(l,[-1])*tf.ones(tf.shape(rho_vec), dtype=rho.dtype)
    t3 = tf.matmul(rho_vec[:,:,tf.newaxis], lvec[:,tf.newaxis,:]) + tf.eye(pdim**2, dtype=rho.dtype)*tf.matmul(tf.expand_dims(lvec,1),rho_vec[:,:,tf.newaxis])

    return t1 + t2 - 2.0*t3

class ZeroSDE:
    '''
    params = [Omega, Gamma, eta] when solving for rho
           = [Omega, Gamma, eta, rho_vec] when solving for I/Q
    '''

    def a(t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)

    def b(t,x,p):
        n = 2 # TODO - Remove hardcoded number of qubits
        return tf.zeros([tf.shape(x)[0],tf.shape(x)[1],n], dtype=x.dtype)

class GenoisSDE:
    '''
    params = [Omega, Gamma, eta] when solving for rho
           = [Omega, Gamma, eta, rho_vec] when solving for I/Q
    '''

    # SDE functions for the density operator
    def a0(t,x,p):
        '''
        Version of a that assumes a rank one parameter tensor
        x - shape = [num_traj,3,1] Upper triangle of rho: [rho(0,0), rho(0,1), rho(1,1)]
        '''
        rho = unwrap_x_to_rho(x[...,0], 2)
        sx, _, sz = paulis()
        ham = -0.5j*tf.cast(p[0], tf.complex128)*(tf.matmul(sx,rho) - tf.matmul(rho,sx))
        supd = supd_herm(tf.pow(0.5*tf.cast(p[1], tf.complex128),0.5)*np.array(sz), rho)

        return wrap_rho_to_x(ham + supd, 2)[:,:,tf.newaxis]

    def b0(t,x,p):
        rho = unwrap_x_to_rho(x[...,0], 2)
        _, _, sz = paulis()
        hi = tf.reshape(wrap_rho_to_x(suph_herm(tf.pow(0.5*tf.cast(p[1], tf.complex128),0.5)*np.array(sz), rho), 2), [-1,3,1])
        hq = tf.reshape(wrap_rho_to_x(suph_herm(-1.0j*tf.pow(0.5*tf.cast(p[1], tf.complex128),0.5)*np.array(sz), rho), 2), [-1,3,1])

        return tf.pow(0.5*tf.cast(p[2], tf.complex128),0.5)*tf.concat([hi, hq], axis=2)

    def bp0(t,x,p):
        # return shape = [num_traj,m=2,d=3,d=3]
        rho = unwrap_x_to_rho(x[...,0], 2)
        _, _, sz = paulis()
        hi = tf.reshape(wrap_rho_to_x(suph_herm_p(tf.pow(0.5*p[1],0.5)*np.array(sz), rho)), [-1,3,1])
        hq = tf.reshape(wrap_rho_to_x(suph_herm_p(-1.0j*tf.pow(0.5*p[1],0.5)*np.array(sz), rho)), [-1,3,1])

        return tf.gather(tf.gather(tf.pow(0.5*p[2],0.5)*tf.concat(hi, hq, axis=1), [0,1,3], axis=2), [0,1,3], axis=3)

    def a(t,x,p):
        '''
        x - shape = [num_traj,3,1] Upper triangle of rho: [rho(0,0), rho(0,1), rho(1,1)]
        '''
        rho = unwrap_x_to_rho(x[...,0], 2)
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]
        sx, _, sz = paulis()
        #ham = -0.5j*tf.cast(p[:,0,tf.newaxis,tf.newaxis], tf.complex128)*(tf.matmul(sx,rho) - tf.matmul(rho,sx))
        pten = -0.5j*tf.cast(tf.expand_dims(tf.dtypes.complex(p[:,0], 0.0), axis=1), dtype=tf.complex128)
        pten = tf.expand_dims(pten, axis=1)
        sx_rho = tf.matmul(sx,rho)
        rho_sx = tf.matmul(rho,sx)
        ham = pten*(sx_rho - rho_sx)
        #supd = supd_herm(tf.pow(0.5*tf.cast(p[:,1,tf.newaxis,tf.newaxis], tf.complex128),0.5)*np.array(sz), rho)
        pten1 = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,1], 0.0),0.5), dtype=tf.complex128)
        pten1 = tf.expand_dims(pten1, axis=1)
        pten1 = tf.expand_dims(pten1, axis=1)
        supd = supd_herm(pten1*np.array(sz), rho)

        return wrap_rho_to_x(ham + supd, 2)[:,:,tf.newaxis]

    def b(t,x,p):
        rho = unwrap_x_to_rho(x[...,0], 2)
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]
        _, _, sz = paulis()
        #hi = tf.reshape(wrap_rho_to_x(suph_herm(tf.pow(0.5*tf.cast(p[:,1,tf.newaxis,tf.newaxis], tf.complex128),0.5)*np.array(sz), rho), 2), [-1,3,1])
        #hq = tf.reshape(wrap_rho_to_x(suph_herm(-1.0j*tf.pow(0.5*tf.cast(p[:,1,tf.newaxis,tf.newaxis], tf.complex128),0.5)*np.array(sz), rho), 2), [-1,3,1])
        pten = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,1], 0.0),0.5), dtype=tf.complex128)
        pten = tf.expand_dims(pten, axis=1)
        pten = tf.expand_dims(pten, axis=1)
        hi = tf.reshape(wrap_rho_to_x(suph_herm(pten*np.array(sz), rho), 2), [-1,3,1])
        hq = tf.reshape(wrap_rho_to_x(suph_herm(-1.0j*pten*np.array(sz), rho), 2), [-1,3,1])

        pten2 = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,2], 0.0),0.5), dtype=tf.complex128)
        pten2 = tf.expand_dims(pten2, axis=1)
        pten2 = tf.expand_dims(pten2, axis=1)
        return pten2*tf.concat([hi, hq], axis=2)

    def bp(t,x,p):
        # return shape = [num_traj,m=2,d=3,d=3]
        rho = unwrap_x_to_rho(x[...,0], 2)
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]
        _, _, sz = paulis()
        pten = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,1], 0.0),0.5), dtype=tf.complex128)
        pten = tf.expand_dims(pten, axis=1)
        pten = tf.expand_dims(pten, axis=1)
        hi = pten*suph_herm_p(np.array(sz), rho)
        hq = -1.0j*pten*suph_herm_p(np.array(sz), rho)
        hi = tf.expand_dims(hi,3)
        hq = tf.expand_dims(hq,3)

        pten2 = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,2], 0.0),0.5), dtype=tf.complex128)
        pten2 = tf.expand_dims(pten2, axis=1)
        pten2 = tf.expand_dims(pten2, axis=1)
        pten2 = tf.expand_dims(pten2, axis=1)

        bp_unwrap = tf.transpose(pten2*tf.concat([hi, hq], axis=3), perm=[0,3,1,2])
        return tf.gather(tf.gather(bp_unwrap, [0,1,3], axis=2), [0,1,3], axis=3)

class GenoisTrajSDE:
    def __init__(self, rhovec, deltat):
        self.rhovec = rhovec
        self.deltat = deltat

    def get_rho(self, t):
        tidx = np.rint(t/self.deltat).astype(int)
        return tf.reshape(self.rhovec[:,:,tidx], [-1,2,2])
    
    def mia0(self,t,x,p):
        rho = self.get_rho(t)
        _, _, sz = paulis()
        l = tf.cast(tf.pow(0.5*p[1],0.5)*np.array(sz), dtype=rho.dtype)
        return tf.cast(tf.pow(0.5*p[2],0.5), dtype=rho.dtype)*tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])

    def mia(self,t,x,p):
        rho = self.get_rho(t)
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]
        _, _, sz = paulis()
        l = tf.cast(tf.pow(0.5*p[:,1,tf.newaxis,tf.newaxis],0.5)*np.array(sz), dtype=rho.dtype)
        return tf.cast(tf.pow(0.5*p[:,2,tf.newaxis,tf.newaxis],0.5), dtype=rho.dtype)*tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])

    def mib(self,t,x,p):
        return tf.ones(tf.shape(x), dtype=x.dtype)

    def mibp(self,t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)

    def mqa(self,t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)

    def mqb(self,t,x,p):
        return tf.ones(tf.shape(x), dtype=x.dtype)

    def mqbp(self,t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)

class FlexSDE:
    '''
    Wrapper class for SDE functions that adds the result to the output of an RNN cell that
    accounts for discrepancies between the assumed model and the true system
    '''

    def __init__(self,a,b,a_cell_real,a_cell_imag,b_cell_real,b_cell_imag):
        self.af = a
        self.bf = b
        self.a_cell_real = a_cell_real
        self.a_cell_imag = a_cell_imag
        self.b_cell_real = b_cell_real
        self.b_cell_imag = b_cell_imag

    def init_states(self, batch_size):
        self.a_state_real = self.a_cell_real.get_initial_state(batch_size=batch_size, dtype=tf.float32)[0]
        self.a_state_imag = self.a_cell_imag.get_initial_state(batch_size=batch_size, dtype=tf.float32)[0]
        self.a_carry_real = self.a_cell_real.get_initial_state(batch_size=batch_size, dtype=tf.float32)[1]
        self.a_carry_imag = self.a_cell_imag.get_initial_state(batch_size=batch_size, dtype=tf.float32)[1]
        self.b_state_real = self.b_cell_real.get_initial_state(batch_size=batch_size, dtype=tf.float32)[0]
        self.b_state_imag = self.b_cell_imag.get_initial_state(batch_size=batch_size, dtype=tf.float32)[0]
        self.b_carry_real = self.b_cell_real.get_initial_state(batch_size=batch_size, dtype=tf.float32)[1]
        self.b_carry_imag = self.b_cell_imag.get_initial_state(batch_size=batch_size, dtype=tf.float32)[1]
    
    def a(self,t,x,p):
        cell_out_real, states = self.a_cell_real(p,[tf.cast(tf.math.real(x[:,:,0]), tf.float32), self.a_carry_real])
        self.a_state_real = states[0]
        self.a_carry_real = states[1]

        cell_out_imag, states = self.a_cell_imag(p,[tf.cast(tf.math.imag(x[:,:,0]), tf.float32), self.a_carry_imag])
        self.a_state_imag = states[0]
        self.a_carry_imag = states[1]

        return self.af(t,x,p) + tf.complex(tf.cast(cell_out_real, tf.float64), tf.cast(cell_out_imag, tf.float64))[:,:,tf.newaxis]

    def b(self,t,x,p):
        cell_out_real, states = self.b_cell_real(p,[tf.cast(tf.math.real(x[:,:,0]), tf.float32), self.b_carry_real])
        self.b_state_real = states[0]
        self.b_carry_real = states[1]

        cell_out_imag, states = self.b_cell_imag(p,[tf.cast(tf.math.imag(x[:,:,0]), tf.float32), self.b_carry_imag])
        self.b_state_imag = states[0]
        self.b_carry_imag = states[1]

        return self.bf(t,x,p) + tf.complex(tf.cast(cell_out_real, tf.float64), tf.cast(cell_out_imag, tf.float64))[:,:,tf.newaxis]

class NNSDE:
    def __init__(self, a_model_real, a_model_imag, b_model_real, b_model_imag):
        self.a_model_real = a_model_real
        self.a_model_imag = a_model_imag
        self.b_model_real = b_model_real
        self.b_model_imag = b_model_imag
    
    def a(self,t,x,p):
        '''
        t - scalar time
        x - shape = [batch_size, num_x]
        p - shape = [batch_size, num_params]
        '''
        xten_real = tf.cast(tf.math.real(x[:,:,0]), tf.float32)
        xten_imag = tf.cast(tf.math.imag(x[:,:,0]), tf.float32)
        tten = t*tf.ones([tf.shape(x)[0],1], xten_real.dtype)
        real_part = self.a_model_real(tf.concat([tten,xten_real,tf.cast(p, xten_real.dtype)], axis=1))
        imag_part = self.a_model_imag(tf.concat([tten,xten_imag,tf.cast(p, xten_real.dtype)], axis=1))

        return tf.complex(tf.cast(real_part, tf.float64), tf.cast(imag_part, tf.float64))[:,:,tf.newaxis]

    def b(self,t,x,p):
        '''
        t - scalar time
        x - shape = [batch_size, num_x]
        p - shape = [batch_size, num_params]
        '''
        xten_real = tf.cast(tf.math.real(x[:,:,0]), tf.float32)
        xten_imag = tf.cast(tf.math.imag(x[:,:,0]), tf.float32)
        tten = t*tf.ones([tf.shape(x)[0],1], xten_real.dtype)
        real_part = self.b_model_real(tf.concat([tten,xten_real,tf.cast(p, xten_real.dtype)], axis=1))
        imag_part = self.b_model_imag(tf.concat([tten,xten_imag,tf.cast(p, xten_real.dtype)], axis=1))

        return tf.complex(tf.cast(real_part, tf.float64), tf.cast(imag_part, tf.float64))[:,:,tf.newaxis]

class RabiWeakMeasSDE:
    '''
    Equations for the stochastic master equation

    drho(t) = -i[H,rho] dt + sum_j=1^n kappa D[Z_j](rho) dt 
                           + sum_j=1^n sqrt(kappa*eta/2) H[Z_j](rho) dW_t^(j)
    
    where
    
    H = sum_j=1^n Omega/2 X_j 
    is the Rabi Hamiltonian
    
    D[L](rho) = L rho L^dagger -(1/2)(L^dagger L rho + rho L^dagger L)
    is the lindblad superoperator
    
    H[L](rho) = L rho + rho L^dagger - rho Tr[rho (L + L^dagger)]
    is the measurement superoperator describing the backction of the weak measurement

    For all a, b, bp,
    params = [Omega, kappa, eta, eps_0, eps_1, ..., eps_k]
    where k can be between 0 and n-1 and eps_j indicates crosstalk between qubits j and j+1
    '''

    def a(t,x,p_in,start_meas=0):
        '''
        x - shape = [num_traj,d=pdim(pdim+1)/2,1] Upper triangle of rho where pdim = 2^n
        return shape = [num_traj,d,1]
        '''
        #pdim = tf.cast(-0.5 + tf.math.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)), dtype=tf.int32)
        #pdim = int(-0.5 + np.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
        #n = np.log2(pdim).astype(int)
        p = p_in
        if t < start_meas:
            p = p*tf.repeat(tf.constant([1,0,0,1], dtype=p.dtype), tf.shape(p)[0])
        
        pdim = 4
        n = 2
        n_choose_2 = 1

        rho = unwrap_x_to_rho(x[...,0], pdim)
        
        #if tf.rank(p) == 1:
        #    p = p[tf.newaxis,:]

        ham = tf.zeros(tf.shape(rho), rho.dtype)

        for j in range(n):
            _, _, _, sxj, _ = [2.0*sm for sm in operations.prod_ops(j, 2, n)]

            #ham = -0.5j*tf.cast(p[:,0,tf.newaxis,tf.newaxis], tf.complex128)*(tf.matmul(sxj,rho) - tf.matmul(rho,sxj))
            pten = -0.5j*tf.cast(tf.expand_dims(tf.dtypes.complex(p[:,0], 0.0), axis=1), dtype=tf.complex128)
            pten = tf.expand_dims(pten, axis=1)
            sx_rho = tf.matmul(sxj,rho)
            rho_sx = tf.matmul(rho,sxj)
            ham = ham + pten*(sx_rho - rho_sx)

        # Add crosstalk terms
        #assert(tf.shape(p)[1] - 3 <= n-1)
        for epsidx in range(3, 3 + n_choose_2):
            _, _, szj, _, syj = [2.0*sm for sm in operations.prod_ops(epsidx - 3, 2, n)]
            _, _, szjp1, _, syjp1 = [2.0*sm for sm in operations.prod_ops(epsidx - 3 + 1, 2, n)]

            pten = -1.0j*tf.cast(tf.expand_dims(tf.dtypes.complex(p[:,epsidx], 0.0), axis=1), dtype=tf.complex128)
            pten = tf.expand_dims(pten, axis=1)
            zz = tf.matmul(szj, szjp1)
            zz_rho = tf.matmul(zz, rho)
            rho_zz = tf.matmul(rho, zz)
            yy = tf.matmul(syj, syjp1)
            yy_rho = tf.matmul(yy, rho)
            rho_yy = tf.matmul(rho, yy)
            ham = ham + pten*(zz_rho - rho_zz)

        x_out = wrap_rho_to_x(ham, pdim)[:,:,tf.newaxis]
        
        for j in range(n):
            _, _, szj, sxj, _ = [2.0*sm for sm in operations.prod_ops(j, 2, n)]

            #supd = supd_herm(tf.pow(0.5*tf.cast(p[:,1,tf.newaxis,tf.newaxis], tf.complex128),0.5)*szj, rho)
            pten1 = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,1], 0.0),0.5), dtype=tf.complex128)
            pten1 = tf.expand_dims(pten1, axis=1)
            pten1 = tf.expand_dims(pten1, axis=1)
            supd = supd_herm(pten1*szj, rho)

            x_out = x_out + wrap_rho_to_x(supd, pdim)[:,:,tf.newaxis]

        return x_out

    def b(t,x,p_in,start_meas=0):
        '''
        x - shape = [num_traj,d=pdim(pdim+1)/2,1] Upper triangle of rho where pdim = 2^n
        return shape = [num_traj,d,m]
        '''
        #pdim = tf.cast(-0.5 + tf.math.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)), dtype=tf.int32)
        #pdim = int(-0.5 + np.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
        #n = np.log2(pdim).astype(int)
        p = p_in
        if t < start_meas:
            p = p*tf.repeat(tf.constant([1,0,0,1], dtype=p.dtype), tf.shape(p)[0])
        
        pdim = 4
        n = 2

        rho = unwrap_x_to_rho(x[...,0], pdim)

        #if tf.rank(p) == 1:
        #    p = p[tf.newaxis,:]

        pten = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,1], 0.0),0.5), dtype=tf.complex128)
        pten = tf.expand_dims(pten, axis=1)
        pten = tf.expand_dims(pten, axis=1)

        pten2 = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,2], 0.0),0.5), dtype=tf.complex128)
        pten2 = tf.expand_dims(pten2, axis=1)
        pten2 = tf.expand_dims(pten2, axis=1)

        x_out = tf.zeros(tf.shape(x), x.dtype)
        for j in range(n):
            _, _, szj, _, _ = [2.0*sm for sm in operations.prod_ops(j, 2, n)]

            #hi = tf.reshape(wrap_rho_to_x(suph_herm(tf.pow(0.5*tf.cast(p[:,1,tf.newaxis,tf.newaxis], tf.complex128),0.5)*szj, rho), 2), [-1,3,1])
            hi = tf.reshape(wrap_rho_to_x(suph_herm(pten*szj, rho), pdim), [-1,tf.shape(x)[1],1])

            if j == 0:
                x_out = pten2*hi
            else:
                x_out = pten2*tf.concat([x_out, hi], axis=2)

        return x_out

    def bp(t,x,p_in,start_meas=0):
        # return shape = [num_traj,m,d,d]
        p = p_in
        if t < start_meas:
            p = p*tf.repeat(tf.constant([1,0,0,1], dtype=p.dtype), tf.shape(p)[0])
        
        pdim = int(-0.5 + np.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
        n = np.log2(pdim).astype(int)

        rho = unwrap_x_to_rho(x[...,0], pdim)
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]
        _, _, sz = paulis()
        pten = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,1], 0.0),0.5), dtype=tf.complex128)
        pten = tf.expand_dims(pten, axis=1)
        pten = tf.expand_dims(pten, axis=1)

        hi = None
        for j in range(n):
            _, _, szj, _, _ = [2.0*sm for sm in operations.prod_ops(j, 2, n)]
            hij = pten*suph_herm_p(np.array(szj), rho)
            hij = tf.expand_dims(hij,3)

            if hi is None:
                hi = hij
            else:
                hi = tf.concat([hi, hij], axis=3)

        pten2 = tf.cast(tf.pow(0.5*tf.dtypes.complex(p[:,2], 0.0),0.5), dtype=tf.complex128)
        pten2 = tf.expand_dims(pten2, axis=1)
        pten2 = tf.expand_dims(pten2, axis=1)
        pten2 = tf.expand_dims(pten2, axis=1)

        bp_unwrap = tf.transpose(pten2*hi, perm=[0,3,1,2])
        vecsize = int(pdim*(pdim+1)/2)
        permidx = tf.concat([[int(pdim*ii + jj) for ii in range(pdim) for jj in range(ii,pdim)], tf.zeros(int(pdim**2 - vecsize), dtype=tf.int32)], axis=0)
        return tf.gather(tf.gather(bp_unwrap, permidx, axis=2), permidx, axis=3)[:,:,:vecsize,:vecsize]

    def get_ham(omega, epsilons, n):
        pdim = 2**n

        ham = np.zeros([pdim, pdim], dtype=np.cdouble)
        for j in range(n):
            _, _, _, sxj, _ = [2.0*sm.numpy() for sm in operations.prod_ops(j, 2, n)]
            ham = ham + 0.5*omega*sxj

        assert(len(epsilons) <= n-1)
        for j, eps in enumerate(epsilons):
            _, _, szj, _, _ = [2.0*sm.numpy() for sm in operations.prod_ops(j, 2, n)]
            _, _, szjp1, _, _ = [2.0*sm.numpy() for sm in operations.prod_ops(j+1, 2, n)]

            ham = ham + eps*np.matmul(szj, szjp1)

        return ham

    def get_liouv(omega, gamma, epsilons, n):
        pdim = 2**n

        liouv = np.zeros([pdim**2, pdim**2], dtype=np.cdouble)
        eye = np.eye(pdim, dtype=np.cdouble)

        # Hamiltonian
        ham = RabiWeakMeasSDE.get_ham(omega, epsilons, n)
        liouv = liouv - 1.0j*(kron(ham, eye).numpy() - kron(eye,np.transpose(ham)).numpy())

        # Lindblad terms
        for j in range(n):
            _, _, szj, _, _ = [2.0*sm.numpy() for sm in operations.prod_ops(j, 2, n)]
            liouv = liouv + 0.5*gamma*(kron(szj, np.transpose(szj)).numpy() - np.eye(pdim**2, dtype=np.cdouble))

        return liouv

class RabiWeakMeasTrajSDE:
    def __init__(self, rhovec, deltat, qidx, start_meas=0):
        '''
        Equations for multi-qubit system voltage records

        rhovec - shape = [num_traj, num_times, pdim, pdim] vectorized density operators
        deltat - time spacing between each time index
        qidx - zero based qubit index
        start_meas - time at which to turn on weak measurement
        '''
        self.pdim = 4
        self.n = 2
        #self.pdim = tf.shape(rhovec)[2]
        #self.n = int(np.math.log2(self.pdim))
        self.rhovec = rhovec
        self.deltat = deltat
        self.qidx = qidx
        self.start_meas = start_meas

    def get_rho(self, t):
        tidx = np.rint(t/self.deltat).astype(int)
        return self.rhovec[:,tidx,:,:]
    
    def mia0(self,t,x,p_in):
        p = p_in
        if t < self.start_meas:
            p = p*tf.repeat(tf.constant([1,0,0,1], dtype=p.dtype), tf.shape(p)[0])
        
        rho = self.get_rho(t)
        _, _, sz = paulis()
        l = tf.cast(tf.pow(0.5*p[1],0.5)*np.array(sz), dtype=rho.dtype)
        return tf.cast(tf.pow(0.5*p[2],0.5), dtype=rho.dtype)*tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])

    def mia(self,t,x,p_in):
        p = p_in
        if t < self.start_meas:
            p = p*tf.repeat(tf.constant([1,0,0,1], dtype=p.dtype), tf.shape(p)[0])
        
        rho = self.get_rho(t)
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]
        _, _, sz = paulis()
        szj = operations.local_op_to_prod(sz, self.qidx, self.n)
        l = tf.cast(tf.pow(0.5*p[:,1,tf.newaxis,tf.newaxis],0.5), dtype=rho.dtype)*szj
        return tf.cast(tf.pow(0.5*p[:,2,tf.newaxis,tf.newaxis],0.5), dtype=rho.dtype)*tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])

    def mib(self,t,x,p):
        if t < self.start_meas:
            return tf.zeros(tf.shape(x), dtype=x.dtype)
        return tf.ones(tf.shape(x), dtype=x.dtype)
    
    def mib_zeros(self,t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)

    def mibp(self,t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)