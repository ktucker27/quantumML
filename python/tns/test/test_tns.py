import numpy as np
import tensorflow as tf
from scipy import optimize
import os
import sys
import unittest

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import networks
import operations
import tns_math

class TestMPS(unittest.TestCase):

    def test_eval(self):
        n = 6
        pdim = 3

        tol = 1e-12

        psi = np.random.uniform(size=pdim**n)
        psi = psi/np.linalg.norm(psi)

        mps = networks.state_to_mps(psi, n, pdim)

        psi2 = mps.state_vector()

        diff = np.linalg.norm(psi - psi2)

        self.assertLessEqual(diff, tol)

    def test_conj(self):
        n = 4
        pdim = 3

        tol = 1e-12

        psi = np.random.uniform(size=pdim**n) + 1j*np.random.uniform(size=pdim**n)
        psi = psi/np.linalg.norm(psi)

        mps = networks.state_to_mps(psi, n, pdim)
        mps = mps.dagger()

        psi2 = mps.state_vector()

        diff = np.linalg.norm(psi - np.conj(psi2))

        self.assertLessEqual(diff, tol)

    def test_inner(self):
        n = 4
        pdim = 2

        tol = 1e-12

        psi = np.random.uniform(size=pdim**n) + 1j*np.random.uniform(size=pdim**n)
        psi2 = np.random.uniform(size=pdim**n) + 1j*np.random.uniform(size=pdim**n)

        mps = networks.state_to_mps(psi, n, pdim)
        mps2 = networks.state_to_mps(psi2, n, pdim)

        val = mps.inner(mps2)

        self.assertLessEqual(abs(np.dot(psi, np.conj(psi2)) - val), tol)

        # Test periodic boundary conditions
        if n >= 4:
            mps2 = mps.substate([1,2])
            psi = mps2.state_vector()
            val = mps2.inner(mps2)

            self.assertLessEqual(abs(np.dot(psi, np.conj(psi)) - val), tol)

    def test_equals(self):
        tol = 1e-12

        n = 4
        pdim = 3

        psi = np.random.uniform(size=pdim**n)
        psi = psi/np.linalg.norm(psi)
        mps = networks.state_to_mps(psi, n, pdim)
        mps2 = mps.substate(range(n))

        self.assertTrue(mps.equals(mps2,tol))

        delta = np.zeros(mps2.tensors[1].shape)
        delta[0,1,0] = 0.5*tol
        mps2.tensors[1] = mps2.tensors[1] + delta
        self.assertTrue(mps.equals(mps2,tol))

        self.assertFalse(mps.equals(mps2, 0.25*tol))

        mps2 = mps.substate(range(1,n-1))
        self.assertFalse(mps.equals(mps2,tol))

    def test_normal(self):
        n = 6
        pdim = 2
        tol = 1e-12

        psi = np.random.uniform(size=pdim**n)
        psi = psi/np.linalg.norm(psi)
        mps = networks.state_to_mps(psi, n, pdim)

        # Initial MPS should be in left normal form
        self.assertTrue(mps.is_left_normal(tol))
        self.assertFalse(mps.is_right_normal(tol))

        # Right normalize and check
        mps.right_normalize()
        self.assertTrue(mps.is_right_normal(tol))
        self.assertFalse(mps.is_left_normal(tol))

        # The result should match the original up to a unit phaser
        psi2 = mps.state_vector()
        phaser = psi[0]/psi2[0]
        self.assertLessEqual(abs(abs(phaser) - 1), tol)
        self.assertLessEqual(np.max(np.abs(psi - phaser*psi2)), tol)

        # Return to left normal
        mps.left_normalize()
        self.assertTrue(mps.is_left_normal(tol))
        self.assertFalse(mps.is_right_normal(tol))

        # The result should match the original up to a unit phaser
        psi2 = mps.state_vector()
        phaser = psi[0]/psi2[0]
        self.assertLessEqual(abs(abs(phaser) - 1), tol)
        self.assertLessEqual(np.max(np.abs(psi - phaser*psi2)), tol)

class TestMPO(unittest.TestCase):
    def test_matrix(self):
        _, _, sz, sx, _ = operations.local_ops(2)
        one_site = [-1.0*sz]
        two_site = [[-1.0*sx,sx]]
        mpo, _ = networks.build_mpo(one_site,two_site,2,4)
        H = mpo.matrix()
        H2 = operations.build_ham(one_site,two_site,2,4)
        self.assertEqual(np.max(np.abs(H - H2)), 0.0)

class TestLocalOps(unittest.TestCase):
    def test_local_ops(self):
        tol = 1e-12

        for n in range(2,11):
            print(f'Checking local ops for N={n}')
            self.check_local_ops(n, tol)

    def check_local_ops(self, n, tol):
        sp, sm, sz, sx, sy = operations.local_ops(n)

        # Test [sp,sm] = 2*sz
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.matmul(sp,sm) - tf.matmul(sm,sp) - 2.0*sz)), tol)

        # Test [sz,sp] = sp
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.matmul(sz,sp) - tf.matmul(sp,sz) - sp)), tol)

        # Test [sz,sm] = -sm
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.matmul(sz,sm) - tf.matmul(sm,sz) + sm)), tol)

        # Test [sx,sy] = 1j*sz
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.matmul(sx,sy) - tf.matmul(sy,sx) - 1.0j*sz)), tol)

        # Test [sy,sz] = 1j*sx
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.matmul(sy,sz) - tf.matmul(sz,sy) - 1.0j*sx)), tol)

        # Test [sz,sx] = 1j*sy
        self.assertLessEqual(tf.reduce_max(tf.abs(tf.matmul(sz,sx) - tf.matmul(sx,sz) - 1.0j*sy)), tol)

class TestPow2Exp(unittest.TestCase):
    def test_pow_2_exp(self):
        rcutoff = 10
        npow = 5
        b = 3
        tol1 = 5e-3
        tol2 = 6e-5
        for p in range(1,11,1):
            print(f'Checking pow_2_exp for p={p}')
            self.check_pow_2_exp(p, rcutoff, b, npow, tol1, tol2)
    
    def check_pow_2_exp(self, p, rcutoff, b, npow, tol1, tol2):
        def f(x,p):
            return x**-p

        x = np.arange(1, rcutoff, 0.1)

        # Check pow_2_exp MSE
        alpha, beta, _ = tns_math.pow_2_exp(p, b, npow)
        pow_2_exp_approx = np.sum((alpha*np.power(np.expand_dims(beta,0),np.expand_dims(x,1))), axis=1)
        mse = np.mean(np.square(f(x,p) - pow_2_exp_approx))
        print(f'Initial mse={mse}')
        self.assertLessEqual(mse, tol1)

        # Check iterative refinement
        ab0 = np.concatenate([alpha,beta])
        fun = lambda alpha_beta : tns_math.exp_loss(alpha_beta, 1.0, p, rcutoff, npow)
        bounds = [[-np.Inf,np.Inf]]*alpha.shape[0] + [[0,np.Inf]]*beta.shape[0]
        result = optimize.minimize(fun, x0=ab0, method='L-BFGS-B', bounds=bounds, options={'maxiter':10000})
        self.assertTrue(result.success)
        ab = result.x
        alpha = ab[:npow]
        beta = ab[npow:]
        opt_approx = np.sum((alpha*np.power(np.expand_dims(beta,0),np.expand_dims(x,1))), axis=1)
        mse = np.mean(np.square(f(x,p) - opt_approx))
        print(f'Final mse={mse}')
        self.assertLessEqual(mse, tol2)

if __name__ == '__main__':
    unittest.main(verbosity=2)