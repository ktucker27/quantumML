import numpy as np
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import networks
import unittest

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

if __name__ == '__main__':
    unittest.main(verbosity=2)