import numpy as np
import tensorflow as tf
import os
import sys
import unittest

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import networks
import operations

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

if __name__ == '__main__':
    unittest.main(verbosity=2)