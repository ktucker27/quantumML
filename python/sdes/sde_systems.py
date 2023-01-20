import math
import numpy as np
import tensorflow as tf

def paulis():
    return [ np.array([[0.0, 1.0],[1.0, 0.0]], dtype=np.cdouble), np.array([[0.0, -1.0j],[1.0j, 0.0]], dtype=np.cdouble), np.array([[1.0, 0.0],[0.0, -1.0]], dtype=np.cdouble) ]

def kron(a,b):
    assert tf.rank(a) == 2
    assert tf.rank(b) == 2
    cp = tf.tensordot(a,b,axes=0)
    c = tf.transpose(cp, perm=[0,2,1,3])
    return tf.reshape(c, [a.shape[0]*b.shape[0], a.shape[1]*b.shape[1]])

def calc_exp(x,o):
    '''
    Input:
    x: shape = [num_traj,4,num_times] set of density operators
    o: shape = [2,2] Hermetian operator to take expectation values of

    Returns:
    exp_o: shape = [num_traj,num_times] operator expectations
    '''

    # Unwrap and permute rho
    rho = tf.transpose(tf.reshape(x, [-1,2,2,x.shape[-1]]), perm=[0,3,1,2])

    return tf.linalg.trace(tf.matmul(rho, o))

class GeometricSDE:
    def a(t,x,p):
        return p[0]*x

    def b(t,x,p):
        return p[1]*x

    def bp(t,x,p):
        return p[1]*tf.ones([x.shape[0],1,1,1])

class Geometric2DSDE:
    def a(t,x,p):
        return tf.reshape((tf.constant([[1.0],[0.0]])*p[0] + tf.constant([[0.0],[1.0]])*p[1]), [1,2,1])*x

    def b(t,x,p):
        return tf.reshape(tf.constant([[1.0, 0.0],[0.0, 0.0]])*p[2] + tf.constant([[0.0, 0.0],[0.0, 1.0]])*p[3],[1,2,2])*x

    def bp(t,x,p):
        num_traj = x.shape[0]
        return tf.tile(tf.expand_dims(tf.stack([tf.constant([[1.0, 0.0],[0.0, 0.0]])*p[2],tf.constant([[0.0, 0.0],[0.0, 1.0]])*p[3]]), axis=0), [num_traj,1,1,1])

class GenoisSDE:
    '''
    params = [Omega, Gamma, eta] when solving for rho
           = [Omega, Gamma, eta, rho_vec] when solving for I/Q
    '''

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
        # return shape = [num_traj,d=4,d=4]
        t1 = tf.map_fn(lambda x: kron(x[0],x[1]), [tf.ones(rho.shape)*l,rho], fn_output_signature=tf.TensorSpec(shape=[4,4]))
        t2 = tf.map_fn(lambda x: kron(x[0],x[1]), [rho,tf.ones(rho.shape)*l], fn_output_signature=tf.TensorSpec(shape=[4,4]))

        lvec = tf.reshape(l,[4])*tf.ones([rho.shape[0],1])
        rho_vec = tf.reshape(rho, [-1,4])
        t3 = tf.tensordot(rho_vec, lvec, axes=0) + tf.linalg.diag(tf.matvec(tf.expand_dims(lvec,1),rho_vec)*tf.ones([1,4]))

        return t1 + t2 - 2.0*t3

    # SDE functions for the density operator
    def a(t,x,p):
        rho = tf.reshape(x, [-1,2,2])
        sx, _, sz = paulis()
        ham = -0.5j*p[0]*(tf.matmul(sx,rho) - tf.matmul(rho,sx))
        supd = GenoisSDE.supd_herm(np.power(0.5*p[1],0.5)*np.array(sz), rho)

        return tf.reshape(ham + supd, [-1,4,1])

    def b(t,x,p):
        rho = tf.reshape(x, [-1,2,2])
        _, _, sz = paulis()
        hi = tf.reshape(GenoisSDE.suph_herm(np.power(0.5*p[1],0.5)*np.array(sz), rho), [-1,4,1])
        hq = tf.reshape(GenoisSDE.suph_herm(-1.0j*np.power(0.5*p[1],0.5)*np.array(sz), rho), [-1,4,1])

        return np.power(0.5*p[2],0.5)*tf.concat([hi, hq], axis=2)

    def bp(t,x,p):
        # return shape = [num_traj,m=2,d=4,d=4]
        rho = tf.reshape(x, [-1,2,2])
        _, _, sz = paulis()
        hi = GenoisSDE.suph_herm_p(np.power(0.5*p[1],0.5)*np.array(sz), rho)
        hq = GenoisSDE.suph_herm_p(-1.0j*np.power(0.5*p[1],0.5)*np.array(sz), rho)
        hi = tf.expand_dims(hi,1)
        hq = tf.expand_dims(hq,1)
        return np.power(0.5*p[2],0.5)*tf.concat(hi, hq, axis=1)

class GenoisTrajSDE:
    def __init__(self, rhovec, deltat):
        self.rhovec = rhovec
        self.deltat = deltat

    def get_rho(self, t):
        tidx = np.rint(t/self.deltat).astype(int)
        return tf.reshape(self.rhovec[:,:,tidx], [-1,2,2])
    
    def mia(self,t,x,p):
        rho = self.get_rho(t)
        _, _, sz = paulis()
        l = np.power(0.5*p[1],0.5)*np.array(sz)
        return np.power(0.5*p[2],0.5)*tf.reshape(tf.linalg.trace(tf.matmul(rho,2.0*l)), [-1,1,1])

    def mib(t,x,p):
        return tf.ones(x.shape)

    def mibp(t,x,p):
        return tf.zeros(x.shape)

    def mqa(t,x,p):
        return tf.zeros(x.shape)

    def mqb(t,x,p):
        return tf.ones(x.shape)

    def mqbp(t,x,p):
        return tf.zeros(x.shape)
