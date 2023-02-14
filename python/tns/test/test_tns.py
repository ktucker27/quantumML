import numpy as np
import tensorflow as tf
import scipy
import os
import sys
import unittest
import time

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import networks
import operations
import tns_math
import tns_solve

# Set verbosity levels
test_tns_verbose = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class TestMPS(unittest.TestCase):

    def test_eval(self):
        n = 6
        pdim = 3

        tol = 1e-12

        psi = np.random.uniform(size=pdim**n)
        psi = psi/np.linalg.norm(psi)

        mps = networks.state_to_mps(psi, n, pdim)

        psi2 = mps.state_vector_eval()

        diff = np.linalg.norm(psi - psi2)

        self.assertLessEqual(diff, tol)

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

    def test_long_range(self):
        tol = 1e-5

        n = 3
        pdim = 7
        rmult = 2
        rpow = 3
        npow = 3

        v = np.zeros([n,n])
        for ii in range(n):
            for jj in range(n):
                if ii != jj:
                    v[ii,jj] = rmult/abs(ii-jj)**rpow

        h = operations.thermal_ham(0, v, pdim, n)

        _, _, sz, sx, sy = operations.local_ops(pdim)
        ops = [[-0.5*sx,sx],[-0.5*sy,sy],[sz,sz]]
        mpo, _ = networks.build_long_range_mpo(ops,pdim,n,rmult,rpow,npow)
        h2 = mpo.matrix()

        self.assertLessEqual(np.max(np.abs(h - h2)), tol)

    def test_oat(self):
        '''
        Tests the construction of the one-axis twisting Hamiltonian
        H = chi*(\sum_i=1^n s_i^z)**2
        '''
        tol = 1e-15

        n = 5
        pdim = 2
        rmult = 2
        rpow = 0
        npow = 3

        # Build the full product space Hamiltonian
        csz = tf.zeros([pdim**n, pdim**n], dtype=tf.complex128)
        for i in range(n):
            _, _, szi, _, _ = operations.prod_ops(i, pdim, n)
            csz = csz + szi
        h = tf.matmul(csz,csz)

        # Build the MPO
        _, _, sz, _, _ = operations.local_ops(pdim)
        ops = [[sz,sz]]
        lops = [0.25*tf.eye(pdim, dtype=tf.complex128)]
        mpo, _ = networks.build_long_range_mpo(ops,pdim,n,rmult,rpow,npow,lops)
        h2 = mpo.matrix()

        self.assertLessEqual(tf.reduce_max(tf.abs(h - h2)), tol)

    def test_purification(self):
        tol = 1e-6

        n = 3
        rpow = 3
        rmult = 2
        npow = 3
        pdim = 2
        _, _, sz, sx, sy = operations.local_ops(pdim)

        # Build the Hamiltonian matrix on the physical/auxiliary product space
        v = np.zeros([n,n])
        for ii in range(n):
            for jj in range(n):
                if ii != jj:
                    v[ii,jj] = rmult/abs(ii-jj)**rpow

        h = operations.thermal_ham(0, v, pdim, n)
        id = tf.eye(pdim**n, dtype=tf.complex128)
        hp = operations.kron(h,id)

        # Build the MPO
        ops = [[-0.5*sx,sx],[-0.5*sy,sy],[sz,sz]]
        mpo, _ = networks.build_purification_mpo(ops,pdim,n,rmult,rpow,npow)

        # Contract the MPO and group to reproduce the matrix
        for ii in range(2*n):
            if ii == 0:
                ten = mpo.tensors[ii]
            else:
                ten = tf.tensordot(ten, mpo.tensors[ii], [[tf.rank(ten)-3],[0]])
        ten = tf.squeeze(ten)

        ten = tf.transpose(ten, perm=(list(range(1,4*n,4)) + list(range(3,4*n,4)) + list(range(0,4*n,4)) + list(range(2,4*n,4))))
        ten = tf.reshape(ten, [(pdim**n)**2, (pdim**n)**2])
        #T2 = T.group({[2*2*n:-4:4,2*2*n-2:-4:2],[2*2*n-1:-4:3,2*2*n-3:-4:1]});

        # Compare the two matrices
        self.assertLessEqual(tf.reduce_max(tf.abs(hp - ten)), tol)

class TestLocalOps(unittest.TestCase):
    def test_local_ops(self):
        tol = 1e-12

        for n in range(2,11):
            if test_tns_verbose: print(f'Checking local ops for N={n}')
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
        tol3 = 1e-3
        for p in range(1,11,1):
            if test_tns_verbose: print(f'Checking pow_2_exp for p={p}')
            self.check_pow_2_exp(p, rcutoff, b, npow, tol1, tol2, tol3)
    
    def check_pow_2_exp(self, p, rcutoff, b, npow, tol1, tol2, tol3):
        def f(x,p):
            return x**-p

        x = np.arange(1, rcutoff, 0.1)

        # Check pow_2_exp MSE
        alpha, beta, _ = tns_math.pow_2_exp(p, b, npow)
        pow_2_exp_approx = np.sum((alpha*np.power(np.expand_dims(beta,0),np.expand_dims(x,1))), axis=1)
        mse = np.mean(np.square(f(x,p) - pow_2_exp_approx))
        if test_tns_verbose: print(f'Initial mse={mse}')
        self.assertLessEqual(mse, tol1)

        # Check iterative refinement
        rmult = 1.0
        alpha, beta, success = tns_math.pow_2_exp_refine(p, b, npow, rmult, rcutoff)
        self.assertTrue(success)
        opt_approx = np.sum((alpha*np.power(np.expand_dims(beta,0),np.expand_dims(x,1))), axis=1)
        mse = np.mean(np.square(rmult*f(x,p) - opt_approx))
        if test_tns_verbose: print(f'Final mse={mse}')
        self.assertLessEqual(mse, tol2)

        # Check iterative refinement with rmult
        rmult = 2.0
        alpha, beta, success = tns_math.pow_2_exp_refine(p, b, npow, rmult, rcutoff)
        self.assertTrue(success)
        opt_approx = np.sum((alpha*np.power(np.expand_dims(beta,0),np.expand_dims(x,1))), axis=1)
        mse = np.mean(np.square(rmult*f(x,p) - opt_approx))
        if test_tns_verbose: print(f'Rmult mse={mse}')
        self.assertLessEqual(mse, tol3)
    
class TestDMRG(unittest.TestCase):

    def test_transverse_ising(self):
        tol = 1e-12
        maxit = 20

        n = 6
        pdim = 2
        jval = 1
        hval = 1

        _, _, sz, sx, _ = operations.local_ops(pdim)
        one_site = [-hval*sz]
        two_site = [[-jval*sx,sx]]

        mpo, _ = networks.build_mpo(one_site,two_site,pdim,n)
        h = mpo.matrix()
        h2 = operations.build_ham(one_site,two_site,pdim,n)
        self.assertLessEqual(tf.reduce_max(tf.abs(h - h2)), tol)

        evals, evecs = tf.linalg.eig(h)
        min_idx = tf.argmin(tf.math.real(evals))
        psi0 = evecs[:,min_idx]
        e0 = evals[min_idx]
        psi = np.random.uniform(size=pdim**n) + 1j*np.random.uniform(size=pdim**n)
        psi = psi/np.linalg.norm(psi)

        mps = networks.state_to_mps(psi, n, pdim)
        mps_out = tns_solve.dmrg(mpo, mps, tol, maxit)
        psi_out = mps_out.state_vector()

        phaser = psi_out[0]/psi0[0]
        self.assertLessEqual(tf.abs(tf.abs(phaser) - 1), tol)

        self.assertLessEqual(tf.reduce_max(tf.abs(psi0 - psi_out/phaser)), tol)

        self.assertLessEqual(abs(mps_out.inner(mps_out) - 1), tol)

        e = mps_out.inner(networks.apply_mpo(mpo, mps_out))
        self.assertLessEqual(abs(e - e0), tol)

class TestTDVP(unittest.TestCase):
    def test_transverse_ising(self):
        debug = True

        tol = 1e-6

        n = 4
        pdim = 2
        J = 1
        h = 1
        dt = 0.01
        tfinal = 1

        _, _, sz, sx, _ = operations.local_ops(pdim)
        one_site = [-h*sz]
        two_site = [[-J*sx,sx]]

        mpo, _ = networks.build_mpo(one_site,two_site,pdim,n)
        H = mpo.matrix()
        H2 = operations.build_ham(one_site,two_site,pdim,n)
        self.assertLessEqual(tf.reduce_max(tf.abs(H - H2)), tol)

        psi0 = np.zeros(pdim**n, dtype=np.cdouble)
        psi0[0] = 1.0
        mps = networks.state_to_mps(psi0, n, pdim)

        t0 = time.time()
        tvec, mps_out, _, _ = tns_solve.tdvp(mpo, mps, dt, tfinal, 0, debug)
        print(f'Run time (s): {time.time() - t0}')

        evec = np.zeros(tvec.shape)
        dtmat = tf.linalg.expm(-1j*H*dt)
        psi = psi0[:,tf.newaxis]
        print(f'Checking {tvec.shape[0]} time values...')
        for ii in range(tvec.shape[0]):
            psi2 = mps_out[ii].state_vector()
            phaser = psi[0,0]/psi2[0]
            self.assertLessEqual(abs(abs(phaser) - 1), tol)
            evec[ii] = tf.reduce_max(tf.abs(psi[:,0] - psi2*phaser))
            psi = tf.matmul(dtmat,psi)

        self.assertLessEqual(tf.reduce_max(evec), tol)

    def test_oat(self):
        debug = True

        tol = 2e-5

        n = 6
        pdim = 2
        chi = 2
        rmult = 2
        rpow = 0
        npow = 3
        dt = 0.01
        tfinal = 1

        # Build the MPO
        _, _, sz, sx, _ = operations.local_ops(pdim)
        ops = [[chi*sz,sz]]
        lops = [(chi*0.25)*tf.eye(pdim, dtype=tf.complex128)]
        mpo, _ = networks.build_long_range_mpo(ops,pdim,n,rmult,rpow,npow,lops)
        mpo_x, _ = networks.build_mpo([sx],[],pdim,n)

        # Build the full product space Hamiltonian
        csx = tf.zeros([pdim**n, pdim**n], dtype=tf.complex128)
        csz = tf.zeros([pdim**n, pdim**n], dtype=tf.complex128)
        for i in range(n):
            _, _, szi, sxi, _ = operations.prod_ops(i, pdim, n)
            csx = csx + sxi
            csz = csz + szi
        H2 = chi*tf.matmul(csz, csz)

        H = mpo.matrix()
        self.assertLessEqual(tf.reduce_max(tf.abs(H - H2)), tol)

        # Get the initial condition +x
        evals, evecs = tf.linalg.eig(csx)
        idx = tf.math.argmax(tf.math.real(evals))
        psi0 = evecs[:,idx]
        mps = networks.state_to_mps(psi0, n, pdim)

        # Do the time evolution
        t0 = time.time()
        tvec, mps_out, _, exp_out = tns_solve.tdvp(mpo, mps, dt, tfinal, 0, debug, None, [mpo_x])
        print(f'Run time (s): {time.time() - t0}')

        # Compare evolved state with the exact state
        evec = np.zeros(tvec.shape)
        dtmat = tf.linalg.expm(-1j*H*dt)
        psi = psi0[:,tf.newaxis]
        print(f'Checking {tvec.shape[0]} time values...')
        for ii in range(tvec.shape[0]):
            psi2 = mps_out[ii].state_vector()
            phaser = psi[0,0]/psi2[0]
            self.assertLessEqual(abs(abs(phaser) - 1), tol)
            evec[ii] = tf.reduce_max(tf.abs(psi[:,0] - psi2*phaser))
            psi = tf.matmul(dtmat, psi)

        self.assertLessEqual(tf.reduce_max(evec), tol)

        # Compare S_x value to expected
        ex = (n/2.0)*np.cos(tvec*chi)**(n-1)
        xerr = tf.reduce_max(tf.abs(ex - exp_out))
        self.assertLessEqual(xerr, tol)

    def test_thermal(self):
        debug = True

        tol = 1e-6

        n = 3
        rpow = 3
        rmult = 2
        npow = 2
        pdim = 2
        _, _, sz, sx, sy = operations.local_ops(pdim)

        # Build the Hamiltonian matrix on the physical/auxiliary product space
        V = np.zeros([n,n], dtype=np.cdouble)
        for ii in range(n):
            for jj in range(n):
                if ii != jj:
                    V[ii,jj] = rmult/abs(ii-jj)**rpow
        H = operations.thermal_ham(0, V, pdim, n)

        # Build the MPO
        ops = [[-0.5*sx,sx],[-0.5*sy,sy],[sz,sz]]
        mpo, _ = networks.build_purification_mpo(ops,pdim,n,rmult,rpow,npow)

        # Perform the imaginary time evolution
        tfinal = -1j
        dt = -0.01*1j
        mps0 = networks.build_init_purification(n,pdim,pdim**n)
        t0 = time.time()
        tvec, _, eout, _ = tns_solve.tdvp(mpo, mps0, dt, tfinal, 0, debug)
        print(f'Run time (s): {time.time() - t0}')

        # Get the exact energy expectations
        eout0 = np.zeros(tvec.shape, dtype=np.cdouble)
        print(f'Checking {tvec.shape[0]} time values...')
        for ii in range(tvec.shape[0]):
            beta = 2*abs(tvec[ii])
            rho = tf.linalg.expm(-beta*H)
            rho = rho/tf.linalg.trace(rho)
            eout0[ii] = tf.linalg.trace(tf.matmul(rho,H))

        # Do the comparison
        self.assertLessEqual(tf.reduce_max(tf.abs(eout - eout0)), tol)

class TestTDVP2(unittest.TestCase):

    def test_transverse_ising(self):
        debug = True

        tol = 1e-6

        n = 4
        pdim = 2
        J = 1
        h = 1
        dt = 0.01
        tfinal = 1
        svdtol = 1e-15

        _, _, sz, sx, _ = operations.local_ops(pdim)
        one_site = [-h*sz]
        two_site = [[-J*sx,sx]]

        mpo, _ = networks.build_mpo(one_site,two_site,pdim,n)
        H = mpo.matrix()
        H2 = operations.build_ham(one_site,two_site,pdim,n)
        self.assertLessEqual(tf.reduce_max(tf.abs(H - H2)), tol)

        # Set up rank one initial condition representing psi0
        psi0 = np.zeros(pdim**n, dtype=np.cdouble)
        psi0[0] = 1.0
        A = np.zeros([1,1,2])
        A[0,0,0] = 1
        ms = []
        for ii in range(n):
            ms.append(tf.constant(A, dtype=tf.complex128))
        mps = networks.MPS(ms)

        t0 = time.time()
        tvec, mps_out, _, _ = tns_solve.tdvp2(mpo, mps, dt, tfinal, svdtol, 0, debug)
        print(f'Run time (s): {time.time() - t0}')

        evec = np.zeros(tvec.shape)
        dtmat = tf.linalg.expm(-1j*H*dt)
        psi = psi0[:,tf.newaxis]
        print(f'Checking {tvec.shape[0]} time values...')
        for ii in range(tvec.shape[0]):
            psi2 = mps_out[ii].state_vector()
            phaser = psi[0,0]/psi2[0]
            self.assertLessEqual(abs(abs(phaser) - 1), tol)
            evec[ii] = tf.reduce_max(tf.abs(psi[:,0] - psi2*phaser))
            psi = tf.matmul(dtmat, psi)

        self.assertLessEqual(tf.reduce_max(evec), tol)

    def test_oat(self):
        debug = True

        tol = 4e-4

        n = 10
        pdim = 2
        chi = 1
        rmult = 2
        rpow = 0
        npow = 3
        dt = 0.01
        tfinal = 1
        svdtol = 1e-6

        # Build the MPO
        _, _, sz, sx, _ = operations.local_ops(pdim)
        ops = [[chi*sz,sz]]
        lops = [(chi*0.25)*tf.eye(pdim, dtype=tf.complex128)]
        mpo, _ = networks.build_long_range_mpo(ops,pdim,n,rmult,rpow,npow,lops)
        mpo_x, _ = networks.build_mpo([sx],[],pdim,n)

        # Build the full product space Hamiltonian
        csx = np.zeros([pdim**n, pdim**n], dtype=float)
        csz = np.zeros([pdim**n, pdim**n], dtype=float)
        for i in range(n):
            _, _, szi, sxi, _ = operations.prod_ops(i, pdim, n)
            csx = csx + tf.cast(sxi, dtype=tf.float64).numpy()
            csz = csz + tf.cast(szi, dtype=tf.float64).numpy()
        H2 = chi*csz*csz

        H = mpo.matrix()
        self.assertLessEqual(tf.reduce_max(tf.abs(H - tf.cast(H2, dtype=tf.complex128))), tol)

        # Get the initial condition +x
        evals, evecs = np.linalg.eig(csx)
        idx = tf.math.argmax(tf.math.real(evals))
        psi0 = evecs[:,idx]
        #mps = networks.state_to_mps(psi0, n, pdim)
        A = tf.ones([1,1,2], dtype=tf.complex128)
        ms = []
        for ii in range(n):
            ms.append(tf.identity(A))
        mps = networks.MPS(ms)

        # Do the time evolution
        t0 = time.time()
        tvec, mps_out, _, exp_out = tns_solve.tdvp2(mpo, mps, dt, tfinal, svdtol, 0, debug, None, [mpo_x])
        print(f'Run time (s): {time.time() - t0}')

        # Compare evolved state with the exact state
        evec = np.zeros(tvec.shape)
        dtmat = scipy.linalg.expm(-1j*H2*dt)
        psi = psi0[:,np.newaxis]
        print(f'Checking {tvec.shape[0]} time values...')
        for ii in range(tvec.shape[0]):
            psi2 = mps_out[ii].state_vector()
            phaser = psi[0,0]/psi2[0]
            self.assertLessEqual(abs(abs(phaser) - 1), 10*tol)
            evec[ii] = tf.reduce_max(tf.abs(psi[:,0] - psi2*phaser))
            psi = np.matmul(dtmat, psi)

        self.assertLessEqual(tf.reduce_max(evec), tol)

        # Compare S_x value to expected
        ex = (n/2.0)*np.cos(tvec*chi)**(n-1)
        xerr = tf.reduce_max(tf.abs(ex - exp_out))
        self.assertLessEqual(xerr, 1e-2)

    def test_thermal_rt(self):
        debug = True

        tol = 5.5e-5

        n = 4
        rpow = 3
        rmult = 1
        npow = 2
        pdim = 2
        dt = 0.01
        tfinal = 1
        svdtol = 1e-15

        _, _, sz, sx, sy = operations.local_ops(pdim)

        # Build the Hamiltonian matrix on the physical/auxiliary product space
        V = np.zeros([n,n])
        csx = np.zeros([pdim**n, pdim**n])
        csy = np.zeros([pdim**n, pdim**n])
        csz = np.zeros([pdim**n, pdim**n])
        for ii in range(n):
            for jj in range(n):
                if ii != jj:
                    V[ii,jj] = rmult/abs(ii-jj)**rpow
            
            _, _, szi, sxi, syi = operations.prod_ops(ii, pdim, n)
            csx = csx + sxi.numpy()
            csy = csy + syi.numpy()
            csz = csz + szi.numpy()
        csx = csx.astype(np.cdouble)
        csz = csz.astype(np.cdouble)
        H = operations.thermal_ham(0, V, pdim, n).numpy()
        s2 = np.matmul(csx,csx) + np.matmul(csy,csy) + np.matmul(csz,csz)

        # Build the MPO
        ops = [[-0.5*sx,sx],[-0.5*sy,sy],[sz,sz]]
        mpo, _ = networks.build_long_range_mpo(ops,pdim,n,rmult,rpow,npow)

        # Build observables to check expectations
        mpo_x, _ = networks.build_mpo([sx],[],pdim,n)

        ops_s = [[sx,sx],[sy,sy],[sz,sz]]
        lops_s = [0.75*tf.eye(pdim, dtype=tf.complex128)]
        mpo_s2, _ = networks.build_long_range_mpo(ops_s,pdim,n,2,0,1,lops_s)

        # Get the initial condition +x
        evals, evecs = np.linalg.eig(csx)
        idx = tf.math.argmax(tf.math.real(evals))
        psi0 = evecs[:,idx]
        #mps = networks.state_to_mps(psi0, n, pdim)
        A = tf.ones([1,1,2], dtype=tf.complex128)
        ms = []
        for ii in range(n):
            ms.append(tf.identity(A))
        mps = networks.MPS(ms)

        t0 = time.time()
        tvec, mps_out, _, exp_out = tns_solve.tdvp2(mpo, mps, dt, tfinal, svdtol, 0, debug, None, [mpo_x, mpo_s2])
        print(f'Run time (s): {time.time() - t0}')

        # Compare evolved state with the exact state
        evec = np.zeros(tvec.shape)
        ex = np.zeros([2,tvec.shape[0]], dtype=np.cdouble)
        dtmat = scipy.linalg.expm(-1j*H*dt)
        psi = psi0[:,np.newaxis]
        print(f'Checking {tvec.shape[0]} time values...')
        for ii in range(tvec.shape[0]):
            psi2 = mps_out[ii].state_vector()
            phaser = psi[0,0]/psi2[0]
            self.assertLessEqual(abs(abs(phaser) - 1), tol)
            evec[ii] = tf.reduce_max(tf.abs(psi[:,0] - psi2*phaser))
            ex[0,ii] = np.matmul(np.matmul(np.transpose(psi).conjugate(), csx), psi)[0][0]
            ex[1,ii] = np.matmul(np.matmul(np.transpose(psi).conjugate(), s2), psi)[0][0]
            psi = np.matmul(dtmat, psi)

        self.assertLessEqual(tf.reduce_max(evec), tol)

        # Compare S_x value to expected
        xerr = tf.reduce_max(tf.abs(ex[0,:] - exp_out[0,:]))
        self.assertLessEqual(xerr, tol)

        # Compare S^2 value to expected
        s2err = tf.reduce_max(tf.abs(ex[1,:] - exp_out[1,:]))
        self.assertLessEqual(s2err, tol)

if __name__ == '__main__':
    unittest.main(verbosity=2)