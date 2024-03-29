import numpy as np
import tensorflow as tf
from scipy import optimize
import operations
import tns_math

class MPS(tf.Module):

    def __init__(self, tensors, eager=False):
        '''
        MPS: Class for representing a matrix product state
             tensors: A cell row vector of rank 3 tensors indexed as follows
                 ___
             0__|   |__1
                |___|
                  |
                  2
        '''
        
        self.tensors = []
        self.eager = eager
        for idx, ten in enumerate(tensors):
            if self.eager:
                self.tensors.append(ten)
            else:
                self.tensors.append(tf.Variable(tf.zeros(tf.shape(ten), ten.dtype), trainable=False))
                self.tensors[idx].assign(ten)
            
        # Validate the incoming tensors
        if self.eager:
            self.validate()

    def validate(self):
        n = len(self.tensors)
        for ii, t in enumerate(self.tensors):
            if tf.rank(t) != 3:
                raise Exception(f'Expected rank 3 tensor at site {ii}')
            
            iim1 = ii-1
            if ii == 0:
                iim1 = n-1
            
            if self.tensors[iim1].shape[1] != t.shape[0]:
                raise Exception(f'Bond dimension mismatch between sites {iim1} and {ii}')

    def assign(self, mps):
        for ii, t in enumerate(self.tensors):
            t.assign(mps.tensors[ii])

    def num_sites(self):
        return len(self.tensors)

    def size(self):
        size = 0
        for ten in self.tensors:
            size += tf.size(ten)

        return size

    def set_tensor(self, ii, t, val=None):
        if val is None:
            val = True
        
        #if tf.rank(t) != 3:
        #    raise Exception(f'Expected rank 3 tensor at site {ii}')
        
        n = self.num_sites()
        
        iim1 = ii-1
        if ii == 0:
            iim1 = n-1
        
        if val and self.tensors[iim1].shape[1] != t.shape[0]:
            raise Exception(f'Bond dimension mismatch between sites {iim1} and {ii}')
        
        iip1 = ii+1
        if ii == n-1:
            iip1 = 0
        
        if val and self.tensors[iip1].shape[0] != t.shape[1]:
            raise Exception(f'Bond dimension mismatch between sites {ii} and {iip1}')

        if self.eager:
            self.tensors[ii] = t
        else:
            self.tensors[ii].assign(t)
    
    def substate(self, indices):
        ms = []
        for idx in indices:
            ms.append(tf.identity(self.tensors[idx]))
        
        return MPS(ms, eager=self.eager)

    def dagger(self):
        ms = []
        for ii in range(self.num_sites()):
            ms.append(tf.math.conj(tf.identity(self.tensors[ii])))

        return MPS(ms, eager=self.eager)

    def dagger_equals(self):
        for ii in range(self.num_sites()):
            self.tensors[ii].assign(tf.math.conj(self.tensors[ii]))

    def inner(self, mps):
        if mps.num_sites() != self.num_sites():
            raise Exception('Inner product attempted between states of different size')
        
        n = self.num_sites()
        ten = tf.tensordot(self.tensors[0], tf.math.conj(mps.tensors[0]), [[2],[2]])
        for ii in range(1,n):
            ten = tf.tensordot(ten, self.tensors[ii], [[1],[0]])
            ten = tf.tensordot(ten, tf.math.conj(mps.tensors[ii]), [[2,4],[0,2]])
            
            if ii != n-1 or tf.rank(ten) > 0:
                ten = tf.transpose(ten, perm=[0,2,1,3])
        
        if tf.rank(ten) != 0:
            ten = operations.trace(ten, [[0,2],[1,3]])
            
            #if tf.rank(ten) != 0:
            #    raise Exception('Expected a scalar at the end of an inner product')
        
        return ten

    def eval(self, sigma):
        if len(sigma) != self.num_sites():
            raise Exception('Index vector has incorrect rank')
        
        psi = self.tensors[0][:,:,sigma[0]]
        for ii in range(1,self.num_sites()):
            psi = tf.matmul(psi, self.tensors[ii][:,:,sigma[ii]])
        
        return operations.trace(psi)

    def rank(self):
        n = self.num_sites()
        r = np.zeros(n-1)
        for ii in range(n-1):
            r[ii] = self.tensors[ii].shape[1]
        
        return r

    def pdim(self):
        n = self.num_sites()
        d = np.zeros(n, dtype=np.int32)
        for ii in range(n):
            d[ii] = self.tensors[ii].shape[2]

        return d
    
    def state_vector(self):
        n = self.num_sites()

        # Contract across all sites
        ten = self.tensors[0]
        for ii in range(1,n):
            ten = tf.tensordot(ten, self.tensors[ii], axes=[[-2], [0]])

        # Trace out the boundary dimensions
        r = tf.rank(ten)
        ten = operations.trace(ten, [[0], [r-2]])

        return tf.reshape(ten, [-1])

    def state_vector_eval(self):
        d = self.pdim()
        psi = np.zeros(np.prod(d), dtype=np.complex128)

        iter = operations.IndexIter(d)
        idx = 0
        while not iter.end():
            psi[idx] = self.eval(iter.curridx)
            idx = idx + 1
            iter.reverse_next()

        return psi

    def equals(self, mps, tol):
        n = self.num_sites()
        if n != mps.num_sites():
            return False
        
        for ii in range(n):
            if not operations.tensor_equal(self.tensors[ii], mps.tensors[ii], tol):
                return False
        
        return True
    
    def is_left_normal(self, tol):
        e = True
        for ii in range(self.num_sites()):
            a = tf.zeros([self.tensors[ii].shape[1], self.tensors[ii].shape[1]], dtype=self.tensors[ii].dtype)
            for jj in range(self.tensors[ii].shape[2]):
                a = a + tf.matmul(tf.transpose(self.tensors[ii][:,:,jj], conjugate=True), self.tensors[ii][:,:,jj])
            
            diff = a - tf.eye(self.tensors[ii].shape[1], dtype=a.dtype)
            if tf.cast(tf.reduce_max(tf.abs(diff)), dtype=tf.float64) > tol:
                e = False
                #break
        
        return e
    
    def is_right_normal(self, tol):
        e = True
        for ii in range(self.num_sites()-1,-1,-1):
            a = tf.zeros([self.tensors[ii].shape[0], self.tensors[ii].shape[0]], dtype=self.tensors[ii].dtype)
            for jj in range(self.tensors[ii].shape[2]):
                a = a + tf.matmul(self.tensors[ii][:,:,jj], tf.transpose(self.tensors[ii][:,:,jj], conjugate=True))
            
            diff = a - tf.eye(self.tensors[ii].shape[0], dtype=a.dtype)
            if tf.cast(tf.reduce_max(tf.abs(diff)), dtype=tf.float64) > tol:
                e = False
                #break
        
        return e
    
    def left_normalize(self, tol=0.0):
        for ii in range(self.num_sites()):
            mdims = self.tensors[ii].shape
            m = tf.reshape(tf.transpose(self.tensors[ii], perm=[0,2,1]), [-1,mdims[1]])
            if tol > 0:
                s, u, v = operations.svd_trunc(m, tol)
            else:
                s, u, v = tf.linalg.svd(m)
            s = tf.cast(tf.linalg.diag(s), dtype=v.dtype)

            ten = tf.transpose(tf.reshape(u, [mdims[0], mdims[2], -1]), perm=[0,2,1])
            if self.eager:
                self.tensors[ii] = ten
            else:
                self.tensors[ii].assign(ten)
            
            # Update the next tensor
            if ii < self.num_sites() - 1:
                next_m = tf.matmul(s, tf.transpose(v, conjugate=True))
                ten = tf.tensordot(next_m, self.tensors[ii+1], [[1],[0]])
                if self.eager:
                    self.tensors[ii+1] = ten
                else:
                    self.tensors[ii+1].assign(ten)
    
    def right_normalize(self, tol=0.0):
        for ii in range(self.num_sites()-1,-1,-1):
            mdims = self.tensors[ii].shape
            m = tf.reshape(self.tensors[ii], [mdims[0],-1])
            if tol > 0:
                s, u, v = operations.svd_trunc(m, tol)
            else:
                s, u, v = tf.linalg.svd(m)
            s = tf.cast(tf.linalg.diag(s), dtype=v.dtype)

            ten = tf.reshape(tf.transpose(v, conjugate=True), [-1,mdims[1],mdims[2]])
            if self.eager:
                self.tensors[ii] = ten
            else:
                self.tensors[ii].assign(ten)
            
            # Update the next tensor
            if ii > 0:
                next_m = tf.matmul(u, s)
                next_m = tf.tensordot(self.tensors[ii-1], next_m, [[1],[0]])
                ten = tf.transpose(next_m, perm=[0,2,1])
                if self.eager:
                    self.tensors[ii-1] = ten
                else:
                    self.tensors[ii-1].assign(ten)
    
    def mps_zeros(n,bdim,pdim,obc):
        '''
        mps_zeros: Build zero Matrix Product State
        
        Parameters:
        n    = Number of sites
        bdim = Bond dimension. If a scalar, will be the bond
               dimension between each pair of consecutive sites.
               Otherwise, bdim(i) is the bond dimension between sites
               i and i + 1
        pdim = Physical dimension. A single scalar to use at each
               site
        obc  = If true, open boundary conditions will be assumed,
               otherwise they will be periodic
        '''
            
        if n < 2:
            raise Exception('MPS requires at least 2 sites')
        
        if tf.rank(bdim) == 1:
            bdim = bdim*tf.ones(n-1)
        elif tf.rank(bdim) != 1 or bdim.shape[0] != n-1:
            raise Exception('bdim must be a scalar or a n-1 vector')
        
        if tf.rank(pdim) == 1:
            pdim = pdim*tf.ones(n)
        elif tf.rank(pdim) != 1 or pdim.shape[0] != n:
            raise Exception('pdim must be a scalar or a n vector')
        
        tensors = []
        for ii in range(n):
            if ii == 0:
                if obc:
                    t = tf.zeros(1,bdim[0],pdim[0])
                else:
                    t = tf.zeros(bdim[n-2],bdim[0],pdim[0])
            elif ii == n-1:
                if obc:
                    t = tf.zeros(bdim[n-2],1,pdim[n-1])
                else:
                    t = tf.zeros(bdim[n-2],bdim[0],pdim[n-1])
            else:
                t = tf.zeros(bdim[ii-1],bdim[ii],pdim[ii])

            tensors.append(t)
        
        return MPS(tensors)

class MPO:
    def __init__(self, tensors):
        '''
        MPO: Class for representing a matrix product operator
        tensors: A cell row vector of rank 4 tensors indexed as follows
             2
            _|_
        0__|   |__1
           |___|
             |
             3
        '''
        
        self.tensors = tensors
        
        # Validate the incoming tensors
        n = len(tensors)
        for ii, t in enumerate(tensors):
            if tf.rank(t) != 4:
                raise Exception(f'Expected rank 4 tensor at site {ii}')
            
            if ii == 0:
                continue
            
            if tensors[ii-1].shape[1] != t.shape[0]:
                raise Exception(f'Bond dimension mismatch between sites {ii-1} and {ii}')

    def num_sites(self):
        return len(self.tensors)

    def eval(self, sigma1, sigma2):
        if len(sigma1) != self.num_sites():
            raise Exception('Index vector 1 has incorrect rank')
        
        if len(sigma2) != self.num_sites():
            raise Exception('Index vector 2 has incorrect rank')
        
        val = self.tensors[0][:,:,sigma2[0],sigma1[0]]
        for ii in range(1,self.num_sites()):
            val = tf.matmul(val, self.tensors[ii][:,:,sigma2[ii],sigma1[ii]])
        
        return operations.trace(val)
    
    def pdim(self):
        '''
        pdim: Returns physical dimensions of the MPO
        pdim(0,:) = Physical dimension of the state operated on
        pdim(1,:) = Physical dimension of the returned state
        '''
        
        d = np.zeros([2, self.num_sites()], dtype=np.int32)
        for ii in range(d.shape[1]):
            d[0,ii] = self.tensors[ii].shape[2]
            d[1,ii] = self.tensors[ii].shape[3]

        return d
    
    def matrix(self):
        n = self.num_sites()
        ten = self.tensors[0]
        for ii in range(1,n):
            ten = tf.tensordot(ten, self.tensors[ii], [[1],[0]])
            ten = tf.transpose(ten, perm=[0, 3, 1, 4, 2, 5])
            ten = tf.reshape(ten, [ten.shape[0], ten.shape[1], ten.shape[2]*ten.shape[3], -1])
        
        return tf.linalg.trace(tf.transpose(ten, perm=[2,3,0,1]))

def state_to_mps(psi, n, pdim):
    assert(len(psi) == pdim**n)

    # The following splitting assumes the convention that the product basis is
    # enumerated with the index on the last site toggling first
    t2 = tf.reshape(psi, [pdim, -1])

    ms = []
    for ii in range(n-1):
        if ii > 0:
            t2 = tf.reshape(c, [c.shape[0]*c.shape[1],-1])
        
        ts, tu, tv = tf.linalg.svd(t2)
        ts = tf.cast(tf.linalg.diag(ts), dtype=tv.dtype)
        
        ms.append(tf.transpose(tf.reshape(tu, [int(tu.shape[0]/pdim), pdim, -1]), perm=[0,2,1]))
        
        tsv_dagger = tf.matmul(ts, tf.transpose(tv, conjugate=True))
        
        if ii < n-2:
            c = tf.reshape(tsv_dagger, [tsv_dagger.shape[0], pdim, int(tsv_dagger.shape[1]/pdim)])
        else:
            ms.append(tf.reshape(tsv_dagger, [tsv_dagger.shape[0],1,-1]))

    return MPS(ms)

def build_mpo(one_site, two_site, pdim, n):
    num_one_site = len(one_site)
    num_two_site = len(two_site)

    d = num_two_site + 2

    m = tf.zeros([d, d, pdim, pdim], dtype=tf.complex128)
    mask = np.zeros([d, d, pdim, pdim], dtype=np.cdouble)
    mask[0,0,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask
    mask[-1,-1,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask

    mask[-1,0,:,:] = 1.0
    for ii in range(num_one_site):
        m = m + mask*one_site[ii]
    mask = 0*mask

    mask2 = np.zeros([d, d, pdim, pdim], dtype=np.cdouble)
    for ii in range(num_two_site):
        mask[1+ii,0,:,:] = 1.0
        mask2[-1,1+ii,:,:] = 1.0
        m = m + mask*two_site[ii][1]
        m = m + mask2*two_site[ii][0]
        mask = 0*mask
        mask2 = 0*mask2

    ms = []
    for ii in range(n):
        if ii == 0:
            ms.append(tf.expand_dims(m[-1,:,:,:],0))
        elif ii == n-1:
            ms.append(tf.expand_dims(m[:,0,:,:],1))
        else:
            ms.append(m)

    return MPO(ms), m

def build_long_range_mpo(ops, pdim, n, rmult, rpow, npow, lops=[]):
    '''
    Implements the finite state automaton approach in section III of
    G. Crosswhite, A. Doherty, G. Vidal
    "Applying matrix product operators to model systems with long-range interactions" (2008)

    Inputs:
      ops - Pairs of operators to be strung together with the form
            (rmult/r**rpow)*(I x ... x I x ops[ii][0] x I^(r-1) x ops[ii][1] x I x ... x I)
            All strings of this type will be summed together to form the MPO
      pdim - Physical dimension
      n - Number of particles
      npow - Number of terms in exponential approximation of r^-rpow
    '''

    # Determine the N term expansion for 1/r^rpow
    if rpow > 0:
        alpha, beta, success = tns_math.pow_2_exp_refine(rpow, 3, npow, 1.0, n)
        assert success
        alpha = alpha*beta # So that coef. are rmult*(sum_n alpha_n*beta_n^(r-1))
                           # One beta always comes with the operator since adjacent
                           # sites => r = 1
    else:
        npow = 1
        alpha = [1.0]
        beta = [1.0]

    # Build the transfer matrix M based on the automaton
    # The operators will be read from right to left, and M(i,j,:,:) is the
    # operator that is applied when transitioning from state j to state i
    num_ops = len(ops)
    d = num_ops*npow + 2
    m = tf.zeros([d, d, pdim, pdim], dtype=tf.complex128)
    mask = np.zeros([d, d, pdim, pdim], dtype=np.cdouble)

    mask[0,0,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask
    mask[-1,-1,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask

    for ii in range(num_ops):
        for jj in range(npow):
            stateidx = 1 + ii*npow + jj
            
            mask[stateidx,0,:,:] = 1.0
            m = m + mask*ops[ii][1]
            mask = 0*mask

            mask[stateidx,stateidx,:,:] = 1.0
            m = m + mask*beta[jj]*np.eye(pdim, dtype=np.cdouble)
            mask = 0*mask

            mask[d-1,stateidx,:,:] = 1.0
            m = m + mask*rmult*alpha[jj]*ops[ii][0]
            mask = 0*mask

    # Add local operators as a direct transition from the first state to the
    # last
    mask[d-1,0,:,:] = 1.0
    for ii in range(len(lops)):
        m = m + mask*lops[ii]
    mask = 0*mask

    # Assemble the MPO
    ms = []
    for ii in range(n):
        if ii == 0:
            ms.append(tf.expand_dims(m[-1,:,:,:],0))
        elif ii == n-1:
            ms.append(tf.expand_dims(m[:,0,:,:],1))
        else:
            ms.append(m)

    return MPO(ms), m

def build_purification_mpo(ops, pdim, n, rmult, rpow, npow, lops=[]):
    '''
    Implements the long-range MPO for a purification of a system P on the purification space
    P x Q where Q is an ancillary space identical to P. The returned operator is H x I in MPO
    form. This is detailed in section 7.2.1 of
    D. Schollwok "The density-matrix renormalization group in the age of matrix product states" (2011)

    Inputs:
      ops - Pairs of operators to be strung together with the form
            (rmult/r**rpow)*(I x ... x I x ops[ii][0] x I^(r-1) x ops[ii][1] x I x ... x I)
            All strings of this type will be summed together to form the MPO
      pdim - Physical dimension
      n - Number of particles
      npow - Number of terms in exponential approximation of r^-rpow
    '''

    # Determine the N term expansion for 1/r^rpow
    if rpow > 0:
        alpha, beta, success = tns_math.pow_2_exp_refine(rpow, 3, npow, 1.0, n)
        assert success
        alpha = alpha*beta # So that coef. are rmult*(sum_n alpha_n*beta_n^(r-1))
                           # One beta always comes with the operator since adjacent
                           # sites => r = 1
    else:
        npow = 1
        alpha = [1.0]
        beta = [1.0]

    # Build the transfer matrix M based on the automaton
    # The operators will be read from right to left, and M(i,j,:,:) is the
    # operator that is applied when transitioning from state j to state i
    num_ops = len(ops)
    d = 2*num_ops*npow + 3
    m = tf.zeros([d, d, pdim, pdim], dtype=tf.complex128)
    mask = np.zeros([d, d, pdim, pdim], dtype=np.cdouble)

    mask[1,0,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask
    mask[0,1,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask
    mask[-1,-1,:,:] = 1.0
    m = m + mask*np.eye(pdim, dtype=np.cdouble)
    mask = 0*mask

    for ii in range(num_ops):
        for jj in range(npow):
            stateidx = 2 + ii*2*npow + jj*2
            
            mask[stateidx,1,:,:] = 1.0
            m = m + mask*ops[ii][1]
            mask = 0*mask

            mask[stateidx+1,stateidx,:,:] = 1.0
            m = m + mask*np.eye(pdim, dtype=np.cdouble)
            mask = 0*mask

            mask[stateidx,stateidx+1,:,:] = 1.0
            m = m + mask*beta[jj]*np.eye(pdim, dtype=np.cdouble)
            mask = 0*mask

            mask[d-1,stateidx+1,:,:] = 1.0
            m = m + mask*rmult*alpha[jj]*ops[ii][0]
            mask = 0*mask

    # Add local operators as a direct transition from the first state to the
    # last
    mask[d-1,1,:,:] = 1.0
    for ii in range(len(lops)):
        m = m + mask*lops[ii]
    mask = 0*mask

    # Assemble the MPO
    ms = []
    for ii in range(2*n):
        if ii == 0:
            ms.append(tf.expand_dims(m[-1,:,:,:],0))
        elif ii == 2*n - 1:
            ms.append(tf.eye(pdim, dtype=tf.complex128)[tf.newaxis,tf.newaxis,:,:])
        elif ii == 2*n - 2:
            ms.append(tf.expand_dims(m[:,1,:,:],1))
        else:
            ms.append(m)

    return MPO(ms), m

def build_init_purification(n, pdim, bond=None):
    if bond is None:
        bond = 1

    # Build a maximally mixed state between a physical site and an auxiliary
    # site
    psi = tf.zeros([pdim**2, 1], dtype=tf.complex128)
    for ii in range(pdim):
        sigma = np.zeros([pdim,1], dtype=np.cdouble)
        sigma[ii,0] = 1
        
        psi = psi + operations.kron(sigma,sigma)
    psi = (1.0/np.sqrt(pdim))*psi

    # Convert to a MPS
    site_mps = state_to_mps(psi, 2, pdim)
    A = site_mps.tensors[0]
    B = site_mps.tensors[1]
    if bond > A.shape[1]:
        T1 = tf.concat([A, tf.zeros([A.shape[0], bond-A.shape[1], A.shape[2]], dtype=A.dtype)], axis=1)
        T2 = tf.concat([B, tf.zeros([bond-B.shape[0], B.shape[1], B.shape[2]], dtype=B.dtype)], axis=0)
    else:
        T1 = A
        T2 = B

    # String the maximally mixed MPS states together, one for each site
    ms = [None for x in range(2*n)]
    for ii in range(n):
        if ii == 0 or bond == 1:
            ms[2*ii] = T1
        else:
            ms[2*ii] = tf.concat([T1, tf.zeros([bond-1, T1.shape[1], T1.shape[2]], dtype=T1.dtype)], axis=0)
        
        if ii == n-1 or bond == 1:
            ms[2*ii+1] = T2
        else:
            ms[2*ii+1] = tf.concat([T2, tf.zeros([T2.shape[0], bond-1, T2.shape[2]], dtype=T2.dtype)], axis=1)

        # Trim excess bond dimension given the position
        trim0 = min([pdim**(2*ii),pdim**(2*n-2*ii)])
        trim1 = min([pdim**(2*ii+1),pdim**(2*n-2*ii-1)])
        trim2 = min([pdim**(2*ii+2),pdim**(2*n-2*ii-2)])
        ms[2*ii] = ms[2*ii][:trim0,:trim1,:]
        ms[2*ii+1] = ms[2*ii+1][:trim1,:trim2,:]

    return MPS(ms)

def apply_mpo(mpo, mps, mpo_mps=None):
    n = mps.num_sites()

    ms = []
    for ii in range(n):
        T = tf.tensordot(mpo.tensors[ii], mps.tensors[ii], [[2],[2]])
        T = tf.transpose(T, perm=[0,3,1,4,2])
        T2 = tf.reshape(T, [T.shape[0]*T.shape[1], T.shape[2]*T.shape[3], T.shape[4]])
        
        ms.append(T2)

    if mpo_mps is None:
        return MPS(ms, eager = mps.eager)
    
    for idx, ten in enumerate(ms):
        mpo_mps.set_tensor(idx, ten, val=False)

    return mpo_mps