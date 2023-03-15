import math
import os
import sys
import numpy as np
import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'tns'))

import operations

def paulis():
    return [ np.array([[0.0, 1.0],[1.0, 0.0]], dtype=np.cdouble), np.array([[0.0, -1.0j],[1.0j, 0.0]], dtype=np.cdouble), np.array([[1.0, 0.0],[0.0, -1.0]], dtype=np.cdouble) ]

def kron(a,b):
    assert tf.rank(a) == 2
    assert tf.rank(b) == 2
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
    t1 = kron(l, tf.eye(2, dtype=rho.dtype))
    t2 = kron(tf.eye(2, dtype=rho.dtype), l)

    rho_vec = tf.reshape(rho, [-1,4])
    lvec = tf.reshape(l,[-1])*tf.ones(tf.shape(rho_vec), dtype=rho.dtype)
    t3 = tf.matmul(rho_vec[:,:,tf.newaxis], lvec[:,tf.newaxis,:]) + tf.eye(4, dtype=rho.dtype)*tf.matmul(tf.expand_dims(lvec,1),rho_vec[:,:,tf.newaxis])

    return t1 + t2 - 2.0*t3

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

    def a(t,x,p):
        '''
        x - shape = [num_traj,d=pdim(pdim+1)/2,1] Upper triangle of rho where pdim = 2^n
        return shape = [num_traj,d,1]
        '''
        #pdim = tf.cast(-0.5 + tf.math.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)), dtype=tf.int32)
        pdim = int(-0.5 + np.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
        n = np.log2(pdim).astype(int)

        rho = unwrap_x_to_rho(x[...,0], pdim)
        
        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]

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
        assert(tf.shape(p)[1] - 3 <= n-1)
        for epsidx in range(3,tf.shape(p)[1]):
            _, _, szj, _, _ = [2.0*sm for sm in operations.prod_ops(epsidx - 3, 2, n)]
            _, _, szjp1, _, _ = [2.0*sm for sm in operations.prod_ops(epsidx - 3 + 1, 2, n)]

            pten = tf.cast(tf.expand_dims(tf.dtypes.complex(p[:,epsidx], 0.0), axis=1), dtype=tf.complex128)
            pten = tf.expand_dims(pten, axis=1)
            ham = ham + pten*tf.matmul(szj, szjp1)

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

    def b(t,x,p):
        '''
        x - shape = [num_traj,d=pdim(pdim+1)/2,1] Upper triangle of rho where pdim = 2^n
        return shape = [num_traj,d,m]
        '''
        #pdim = tf.cast(-0.5 + tf.math.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)), dtype=tf.int32)
        pdim = int(-0.5 + np.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
        n = np.log2(pdim).astype(int)

        rho = unwrap_x_to_rho(x[...,0], pdim)

        if tf.rank(p) == 1:
            p = p[tf.newaxis,:]

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

    def bp(t,x,p):
        # return shape = [num_traj,m,d,d]
        pdim = int(-0.5 + np.sqrt(0.25 + 2.0*tf.cast(tf.shape(x)[1], dtype=tf.float32)))
        n = np.log2(pdim).astype(int)
        
        rho = unwrap_x_to_rho(x[...,0], pdim)
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

class RabiWeakMeasTrajSDE:
    def __init__(self, rhovec, deltat, qidx):
        '''
        Equations for multi-qubit system voltage records

        rhovec - shape = [num_traj, num_times, pdim, pdim] vectorized density operators
        deltat - time spacing between each time index
        qidx - zero based qubit index
        '''
        self.pdim = tf.shape(rhovec)[2]
        self.n = int(np.math.log2(self.pdim))
        self.rhovec = rhovec
        self.deltat = deltat
        self.qidx = qidx

    def get_rho(self, t):
        tidx = np.rint(t/self.deltat).astype(int)
        return self.rhovec[:,tidx,:,:]
    
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
        szj = tf.cast(operations.local_op_to_prod(sz, self.qidx, self.n), dtype=p.dtype)
        l = tf.cast(tf.pow(0.5*p[:,1,tf.newaxis,tf.newaxis],0.5)*szj, dtype=rho.dtype)
        return tf.cast(tf.pow(0.5*p[:,2,tf.newaxis,tf.newaxis],0.5), dtype=rho.dtype)*tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])

    def mib(self,t,x,p):
        return tf.ones(tf.shape(x), dtype=x.dtype)

    def mibp(self,t,x,p):
        return tf.zeros(tf.shape(x), dtype=x.dtype)
