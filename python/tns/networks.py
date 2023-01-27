import numpy as np
import tensorflow as tf
import operations

class MPS:

    def __init__(self, tensors):
        '''
        MPS: Class for representing a matrix product state
             tensors: A cell row vector of rank 3 tensors indexed as follows
                 ___
             0__|   |__1
                |___|
                  |
                  2
        '''
        
        self.tensors = tensors
            
        # Validate the incoming tensors
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

    def num_sites(self):
        return len(self.tensors)

    def set_tensor(self, ii, t, val=None):
        if val is None:
            val = True
        
        if tf.rank(t) != 3:
            raise Exception(f'Expected rank 3 tensor at site {ii}')
        
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
        
        self.tensors[ii] = t
    
    def substate(self, indices):
        ms = []
        for idx in indices:
            ms.append(tf.identity(self.tensors[idx]))
        
        return MPS(ms)

    def dagger(self):
        ms = []
        for ii in range(self.num_sites()):
            ms.append(tf.math.conj(tf.identity(self.tensors[ii])))

        return MPS(ms)

    def inner(self, mps):
        if mps.num_sites() != self.num_sites():
            raise Exception('Inner product attempted between states of different size')
        
        mps = mps.dagger()
        
        n = self.num_sites()
        ten = tf.tensordot(self.tensors[0], mps.tensors[0], [[2],[2]])
        for ii in range(1,n):
            ten = tf.tensordot(ten, self.tensors[ii], [[1],[0]])
            ten = tf.tensordot(ten, mps.tensors[ii], [[2,4],[0,2]])
            
            if ii != n-1 or tf.rank(ten) > 0:
                ten = tf.transpose(ten, perm=[0,2,1,3])
        
        if tf.rank(ten) != 0:
            ten = operations.trace(ten, [[0,2],[1,3]])
            
            if tf.rank(ten) != 0:
                raise Exception('Expected a scalar at the end of an inner product')
        
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
        r = tf.zeros(n-1)
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
                break
        
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
                break
        
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

            self.tensors[ii] = tf.transpose(tf.reshape(u, [mdims[0], mdims[2], -1]), perm=[0,2,1])
            
            # Update the next tensor
            if ii < self.num_sites() - 1:
                next_m = tf.matmul(s, tf.transpose(v, conjugate=True))
                self.tensors[ii+1] = tf.tensordot(next_m, self.tensors[ii+1], [[1],[0]])
    
    def right_normalize(self, tol=0.0):
        for ii in range(self.num_sites()-1,-1,-1):
            mdims = self.tensors[ii].shape
            m = tf.reshape(self.tensors[ii], [mdims[0],-1])
            if tol > 0:
                s, u, v = operations.svd_trunc(m, tol)
            else:
                s, u, v = tf.linalg.svd(m)
            s = tf.cast(tf.linalg.diag(s), dtype=v.dtype)

            self.tensors[ii] = tf.reshape(tf.transpose(v, conjugate=True), [-1,mdims[1],mdims[2]])
            
            # Update the next tensor
            if ii > 0:
                next_m = tf.matmul(u, s)
                next_m = tf.tensordot(self.tensors[ii-1], next_m, [[1],[0]])
                self.tensors[ii-1] = tf.transpose(next_m, perm=[0,2,1])
    
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
            
        d = tf.zeros([2, self.num_sites()])
        for ii in range(d.shape[1]):
            d[0,ii] = self.tensors[ii].shape[2]
            d[1,ii] = self.tensors[ii].shape[3]
    
    def matrix(self):
        pdim = self.pdim()
        
        iter1 = operations.IndexIter(pdim[1,:])
        op = tf.zeros(tf.reduce_prod(pdim[1,:]),tf.reduce_prod(pdim[0,:]))
        ii = 0
        while not iter1.end():
            iter2 = operations.IndexIter(pdim[0,:])
            jj = 0
            while not iter2.end():
                op[ii,jj] = self.eval(iter1.curridx, iter2.curridx)
                jj = jj + 1
                iter2.reverse_next()

            ii = ii + 1
            iter1.reverse_next()

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
