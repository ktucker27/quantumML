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
        if s[...,ii,ii] > tol:
            endidx = endidx + 1
        else:
            break
    
    if maxrank is not None:
        endidx = min([endidx,maxrank])
    
    return s[...,:endidx,:endidx], u[...,:endidx], v[...,:endidx]

class IndexIter:

    def __init__(self,dim):
        self.rank = len(dim)
        self.curridx = np.zeros(self.rank)
        self.endidx = -1*np.ones(self.rank)
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

    