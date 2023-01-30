import numpy as np
import tensorflow as tf

def trace(ten, indices=None):
    '''
    Returns the contraction of a tensor within itself (i.e. a trace) for indices
    in the corresponding list
    indices[0][i] and indices[1][i] will be contracted for each i
    If indices is not provided, all adjacent indices will be contracted
    '''

    if indices is None:
        indices = [[],[]]
        for ii in range(0,tf.rank(ten),2):
            indices[0].append(ii)
            indices[1].append(ii+1)

    perm = []
    for ii in range(tf.rank(ten)):
        if ii in indices[0] or ii in indices[1]:
            continue
        perm.append(ii)

    for idx1, idx2 in zip(indices[0], indices[1]):
        if ten.shape[idx1] != ten.shape[idx2]:
            raise Exception(f'trace: Indices {idx1} and {idx2} have different dimensions')
        
        perm.append(idx1)
        perm.append(idx2)

    ten = tf.transpose(ten, perm=perm)

    for ii in range(len(indices[0])):
        ten = tf.linalg.trace(ten)

    return ten

def tensor_equal(a,b,tol=0.0):
    if not np.array_equal(a.shape, b.shape):
        return False
    
    return tf.cast(tf.reduce_max(tf.abs(a - b)), dtype=tf.float64) <= tol

def svd_trunc(a,tol=0.0,maxrank=None):
    s, u, v = tf.linalg.svd(a)

    endidx = 0
    for ii in range(a.shape[0]):
        if s[ii] > tol:
            endidx = endidx + 1
        else:
            break
    
    if maxrank is not None:
        endidx = min([endidx,maxrank])
    
    return s[:endidx], u[...,:endidx], v[...,:endidx]

def kron(a,b):
    assert tf.rank(a) == 2
    assert tf.rank(b) == 2
    cp = tf.tensordot(a,b,axes=0)
    c = tf.transpose(cp, perm=[0,2,1,3])
    return tf.reshape(c, [a.shape[0]*b.shape[0], a.shape[1]*b.shape[1]])

def local_ops(n):
    '''
    Returns nxn spin-(n-1)/2 operators on a single site in the following order:
    [sp, sm, sx, sy, sz]
    '''
    ji = (n-1)/2.0
    mvec = np.arange(ji,-ji-1.0,-1.0, dtype=np.cdouble)
    
    sz = tf.linalg.diag(mvec)
    sp = tf.roll(tf.sqrt(tf.linalg.diag((ji - mvec)*(ji + mvec + 1))), shift=-1, axis=0)
    sm = tf.roll(tf.sqrt(tf.linalg.diag((ji + mvec)*(ji - mvec + 1))), shift=1, axis=0)
    sx = 0.5*(sp + sm)
    sy = -0.5j*(sp - sm)

    return sp, sm, sz, sx, sy

def local_op_to_prod(olocal, idx, n):
    pdim = olocal.shape[0]
    idn = tf.eye(pdim, dtype=olocal.dtype)
    id = tf.ones([1,1], dtype=olocal.dtype)
    for ii in range(idx):
        id = kron(id, idn)

    o = kron(id, olocal)

    id = tf.ones([1,1], dtype=olocal.dtype)
    for ii in range(idx+1,n):
        id = kron(id, idn)

    return kron(o, id)

def build_ham(one_site,two_site,pdim,n):
    '''
    Builds a Hamiltonian with the given pdim x pdim single site operators and pairs
    of two site operators for each of the n sites, summed together
    '''
    h = tf.zeros([pdim**n, pdim**n], dtype=tf.complex128)
    for ii in range(n):
        for jj in range(len(one_site)):
            h = h + local_op_to_prod(one_site[jj], ii, n)
        
        if ii < n - 1:
            for jj in range(len(two_site)):
                a = local_op_to_prod(two_site[jj][0], ii, n)
                b = local_op_to_prod(two_site[jj][1], ii+1, n)
                h = h + tf.matmul(a,b)

    return h

class IndexIter:

    def __init__(self,dim):
        self.rank = len(dim)
        self.curridx = np.zeros(self.rank, dtype=np.int32)
        self.endidx = -1*np.ones(self.rank, dtype=np.int32)
        self.dim = dim
    
    def end(self):
        return np.array_equal(self.curridx, self.endidx)

    def equals(self,iter):
        return np.array_equal(self.curridx, iter.curridx)

    def next(self):
        if self.end():
            return

        for ii in range(self.rank):
            self.curridx[ii] = self.curridx[ii] + 1
            if self.curridx[ii] < self.dim[ii]:
                break

            if ii == self.rank - 1:
                self.curridx = self.endidx
            else:
                self.curridx[ii] = 0
    
    def reverse_next(self):
        if self.end():
            return

        for ii in range(self.rank-1,-1,-1):
            self.curridx[ii] = self.curridx[ii] + 1
            if self.curridx[ii] < self.dim[ii]:
                break

            if ii == 0:
                self.curridx = self.endidx
            else:
                self.curridx[ii] = 0

    