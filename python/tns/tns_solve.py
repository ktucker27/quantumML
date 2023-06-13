import tensorflow as tf
import numpy as np
import networks
import operations

def lanczos_mps(L, R, ops, b, numsteps):
    '''
    An efficient Lanczos routine for common matrix vector operations in MPS algorithms

    Can be found as algorithm 6.10 in section 6.6.1 of
    J. Demmel "Applied Numerical Linear Algebra" (1997)

    Works by finding the orthonormal basis of the Krylov space containing b such that
    Q^T A Q = T
    where T is tridiagonal with alpha on the diagonal and beta above and below. The first
    column of Q is b normalized

    L and R are the tensors defining the symmetric matrix A when contracted together with
    the MPO tensors in ops. The length of ops is the number of sites incorporated into b

    b is the tensor being operated on with the following indices:

    Zero site:
        ___
    0__|   |__1
       |___|

    One site:
        ___
    0__|   |__1
       |___|
         |
         2

    Two site:
        ______
    0__|      |__1
       |______|
         |  |
         2  3
    
    Returns: T, Q, alphas, betas
    '''
    num_sites = len(ops)
    bvec = tf.reshape(b, [-1])
    qmat = bvec/tf.linalg.norm(bvec)
    qmat = tf.reshape(qmat, [-1,1])
    alphas = tf.zeros([1], dtype=tf.complex128)
    betas = tf.zeros([1], dtype=tf.complex128)
    ab_set = False
    alpha = tf.constant(0.0, dtype=tf.complex128)
    beta = tf.constant(0.0, dtype=tf.complex128)
    for i in tf.range(numsteps+1):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(qmat, tf.TensorShape([None, None])),
                              (alphas, tf.TensorShape([None])),
                              (betas, tf.TensorShape([None])),
                              (b, tf.TensorShape(None))]
        )
        # In the following, order of operations are given in terms of the
        # dimensions:
        #
        # m1 = left bond dimension
        # m2 = right bond dimension
        #  k = MPO bond dimension
        #  d = physical dimension
        #  N = number of sites in b (1 or 2)
        
        # Perform the matrix vector multiplication with an efficient bubbling
        # z = A*Q(:,i); (Regular Lanczos analog)
        b = tf.tensordot(L, b, [[0],[0]]) # O(m1^2 m2 k d^N)
        
        if num_sites == 0:
            b = tf.tensordot(b, R, [[2,0],[0,1]]) # O(m1 m2^2 k)
        elif num_sites == 1:
            b = tf.tensordot(b, ops[0], [[0,3],[0,2]]) # O(m1 m2 k^2 d^2)
            b = tf.tensordot(b, R, [[1,2],[0,1]]) # O(m1 m2^2 k d)
        else:
            b = tf.tensordot(b, ops[0], [[0,3],[0,2]]) # O(m1 m2 k^2 d^3)
            b = tf.tensordot(b, ops[1], [[3,2],[0,2]]) # O(m1 m2 k^2 d^3)
            b = tf.tensordot(b, R, [[1,3],[0,1]]) # O(m1 m2^2 k d^2)
        
        if num_sites == 1:
            b = tf.transpose(b, perm=[0,2,1])
        elif num_sites == 2:
            b = tf.transpose(b, perm=[0,3,1,2])
        
        bdims = tf.shape(b)
        z = tf.reshape(b, [-1])
        alpha = tf.squeeze(tf.matmul(tf.math.conj(qmat[:,i][tf.newaxis,:]), z[:,tf.newaxis]))
        
        if i == numsteps:
            break
        
        z = z - alpha*qmat[:,i]
        if i > 0:
            z = z - beta*qmat[:,i-1]
        beta = tf.cast(tf.linalg.norm(z), dtype=tf.complex128)
        if abs(beta) < 1e-10:
            #print(f'WARNING - Found linear dependence on step {i}')
            numsteps = i
            break
        qmat = tf.concat((qmat, z[:,tf.newaxis]/beta), axis=1)

        ab_set = True
        if i == 0:
            alphas = tf.ones([1], dtype=tf.complex128)*alpha
        else:
            alphas = tf.concat((alphas, [alpha]), axis=0)

        if i == 0:
            betas = tf.ones([1], dtype=tf.complex128)*beta
        else:
            betas = tf.concat((betas, [beta]), axis=0)
        
        b = qmat[:,i+1]
        b = tf.reshape(b, bdims)

    if not ab_set:
        alphas = tf.ones([1], dtype=tf.complex128)*alpha
    else:
        alphas = tf.concat((alphas, [alpha]), axis=0)

    if ab_set:
        beta_mat = tf.linalg.diag(tf.concat(([0.0],betas), axis=0))
        sup = tf.roll(beta_mat, shift=-1, axis=0)
        sub = tf.roll(beta_mat, shift=-1, axis=1)
        tmat = tf.linalg.diag(alphas) + sup + sub
    else:
        tmat = tf.zeros([1,1], dtype=tf.complex128)

    return tmat, qmat, alphas, betas

def lanczos_expm_mps(L, R, ops, v, m, fun=None):
    '''
    An approximation for a general function of a matrix times a vector in MPS form.
    Based on equation (2.7) in
    M. Hochbruck, C. Lubich, "On Krylov Subspace Approximations to the Matrix Exponential Operator" (1997)
    '''
    if fun is None:
        fun = lambda x: tf.exp(x)

    #n = tf.reduce_prod(v.shape)

    #if m >= n:
    #    raise Exception('Need m < n')
        
    tmat, qmat, _, _ = lanczos_mps(L, R, ops, v, m)
        
    evals, evec = tf.linalg.eig(tmat)

    return tf.squeeze(tf.matmul(qmat, tf.matmul(evec, tf.matmul(tf.linalg.diag(fun(evals)), tf.math.conj(evec[0,:][:,tf.newaxis])))))

def dmrg(mpo, mps, tol, maxit, eps=1e-12):
    '''
    Implements the iterative DMRG algorithm given in section 6.3 of
    D. Schollwok "The density-matrix renormalization group in the age of matrix product states" (2011)
    See in particular the bulleted optimal algorithm 

    Inputs:
    mpo - The Hamiltonian as an MPO
    mps - The initial MPS
    tol - Iteration will stop once the energy variance is below this tolerance
    maxit - The maximum number of allowed iterations
    eps - Allowed error for normalization of the MPS

    Returns:
    mps_out - Ground state estimate for the Hamiltonian as an MPS
    '''

    n = mps.num_sites()

    if not mps.is_right_normal(eps):
        mps.right_normalize()
    
    ms = mps.substate(range(n))
    msd = mps.dagger()

    # Initialize the R list
    R = [tf.ones([1,1,1], dtype=tf.complex128)]
    ten = tf.tensordot(mpo.tensors[n-1], ms.tensors[n-1], [[2],[2]])
    ten = tf.tensordot(ten, msd.tensors[n-1], [[2],[2]])
    R.insert(0, tf.transpose(ten, perm=[2,0,4,3,1,5]))
    for ii in range(n-3,-1,-1):
        ten = tf.tensordot(R[0], ms.tensors[ii+1], [[0],[1]])
        ten = tf.tensordot(ten, mpo.tensors[ii+1], [[0,6],[1,2]])
        ten = tf.tensordot(ten, msd.tensors[ii+1], [[0,6],[1,2]])
        R.insert(0, tf.transpose(ten, perm = [3,4,5,0,1,2]))

    # Initialize the L list
    L = [tf.ones([1,1,1], dtype=tf.complex128)]
    for ii in range(n-1):
        L.append(None)

    startidx = 0
    idxinc = 1
    endidx = n-2
    
    for itidx in range(2*maxit):
        # Sweep to the right/left updating tensors
        for ii in range(startidx, endidx+idxinc, idxinc):
            nextidx = ii + idxinc
            
            # Contract the MPO state with R
            TR = R[ii]
            if ii < n-1:
                while TR.shape[-1] == 1:
                    TR = tf.squeeze(TR, axis=-1)
            A = tf.tensordot(mpo.tensors[ii], TR, [[1],[1]])
            
            # Contract the result with L
            TL = L[ii]
            if ii > 0:
                while TL.shape[-1] == 1:
                    TL = tf.squeeze(TL, axis=-1)
            A = tf.tensordot(A, TL, [[0],[1]])
            
            # Group the tensor into a matrix and get the eigenvector
            alldims = tf.shape(A)
            mdims = [alldims[4], alldims[2], alldims[0]]
            A = tf.transpose(A, perm=[5,3,1,4,2,0])
            A = tf.reshape(A, [-1,np.prod(mdims)])
            evals, evec = tf.linalg.eig(A)
            idx = tf.math.argmin(tf.math.real(evals))
            M = evec[:,idx] # TODO - Find a more efficient way of getting this
            if tf.rank(M) != 1:
                raise Exception(f'Expected rank one tensor from eigenvector, got {tf.rank(M)}')
            M2 = tf.reshape(M, mdims)

            # Update the tensors
            if idxinc > 0:
                # Left normalize
                M2 = tf.transpose(M2, perm=[0,2,1])
                M2 = tf.reshape(M2, [-1, M2.shape[2]])
                ts, tu, tv = tf.linalg.svd(M2)
                ts = tf.cast(tf.linalg.diag(ts), dtype=M2.dtype)
                new_m = tf.reshape(tu, [mdims[0], mdims[2], mdims[1]])
                new_m = tf.transpose(new_m, perm=[0,2,1])
                
                # Update the next tensor
                next_m = tf.matmul(ts, tf.transpose(tv, conjugate=True))
                next_m = tf.tensordot(next_m, ms.tensors[nextidx],[[1],[0]])
            else:
                # Right normalize
                M2 = tf.reshape(M2, [M2.shape[0], -1])
                ts, tu, tv = tf.linalg.svd(M2)
                ts = tf.cast(tf.linalg.diag(ts), dtype=M2.dtype)
                new_m = tf.transpose(tv, conjugate=True)
                new_m = tf.reshape(new_m, mdims)
                
                # Update the next tensor
                next_m = tf.matmul(tu, ts)
                next_m = tf.tensordot(ms.tensors[nextidx], next_m, [[1],[0]])
                next_m = tf.transpose(next_m, perm=[0,2,1])
            
            ms.set_tensor(ii, new_m)
            msd.set_tensor(ii, tf.math.conj(new_m))
            
            ms.set_tensor(nextidx, next_m)
            msd.set_tensor(nextidx, tf.math.conj(next_m))

            if idxinc > 0:
                if ii == 0:
                    # Initialize the L list
                    T = tf.tensordot(mpo.tensors[0], ms.tensors[0], [[2],[2]])
                    T = tf.tensordot(T, msd.tensors[0], [[2],[2]])
                    L[ii+1] = tf.transpose(T, perm=[3,1,5,2,0,4])
                else:
                    # Update the L list
                    T = tf.tensordot(L[ii], ms.tensors[ii], [[0],[0]])
                    T = tf.tensordot(T, mpo.tensors[ii], [[0,6],[0,2]])
                    T = tf.tensordot(T, msd.tensors[ii], [[0,6],[0,2]])
                    L[ii+1] = tf.transpose(T, perm=[3,4,5,0,1,2])
            else:
                if ii == n-1:
                    # Initialize the R list
                    T = tf.tensordot(mpo.tensors[n-1], ms.tensors[n-1], [[2],[2]])
                    T = tf.tensordot(T, msd.tensors[n-1], [[2],[2]])
                    R[ii-1] = tf.transpose(T, perm=[2,0,4,3,1,5])
                else:
                    # Update the R list
                    T = tf.tensordot(R[ii], ms.tensors[ii], [[0],[1]])
                    T = tf.tensordot(T, mpo.tensors[ii], [[0,6],[1,2]])
                    T = tf.tensordot(T, msd.tensors[ii], [[0,6],[1,2]])
                    R[ii-1] = tf.transpose(T, perm=[3,4,5,0,1,2])

        # Compute energy variance to see if we've converged
        if idxinc < 0:
            mpo_ms = networks.apply_mpo(mpo, ms)
            mpo_mpo_ms = networks.apply_mpo(mpo, mpo_ms)
            var = ms.inner(mpo_mpo_ms) - (ms.inner(mpo_ms))**2

            if tf.reduce_max(tf.abs(tf.math.real(var))) > eps:
                raise Exception('Found complex energy variance')
            var = tf.math.real(var)

            if var < -eps:
                raise Exception('Found negative energy variance')

            if var < tol:
                print(f'Converged in {(itidx+1)/2.0} iterations with energy variance {var}')
                break
        
        # Flip the sweep direction
        if idxinc > 0:
            startidx = n-1
            idxinc = -1
            endidx = 1
        else:
            startidx = 0
            idxinc = 1
            endidx = n-2
            
        if itidx >= 2*maxit:
            print('WARNING - Max iterations reached')

    return networks.MPS(ms.tensors)

#@tf.function
def tdvp(mpo, mps, dt, tfinal, eps=0.0, debug=False, ef=None, exp_ops=[], anc_mps=[None, None], anc_mpo_mps=[], mps_out=[]):
    '''
    An implementation of the time-dependent variational principle algortihm found in
    J. Haegeman, F. Verstraete, et. al "Unifying time evolution and optimization with matrix product states" (2015)
    See in particular equations (5) and (6), as well as the discussion immediately following

    Inputs:
    mpo - The Hamiltonian as an MPO
    mps - The initial MPS
    dt - Time step
    tfinal - The total time over which to evolve the state
    eps - Allowed error for normalization of the MPS
    debug - Allow verbose output if true
    ef - If provided, evolution will stop once this energy threshold is crossed
    exp_ops - List of operators in MPO form to perform expected value calculations for at each time step
    anc_mps - Two ancillary MPS objects, needed when running in graph mode
    anc_mpo_mps - Ancillary MPS objects, one for the energy plus one for each exp_ops, needed when running in graph mode
    mps_out - List of MPS states at each time

    Returns:
    tvec - Vector of times
    eout - Expected energy values at each time
    exp_out - [len(exp_ops), num_times] matrix of expected operator values
    '''
    tol = 1e-12

    ms = anc_mps[0]
    msd = anc_mps[1]

    if len(anc_mpo_mps) == 0:
        [anc_mpo_mps.append(None) for _ in range(len(exp_ops)+1)]

    numt = int(abs(tfinal)/abs(dt) + 1)
    tvec = np.zeros(numt, dtype=np.array(dt).dtype)
    if len(mps_out) == 0:
        [mps_out.append(None) for _ in range(numt)]
    exp_out = tf.zeros([len(exp_ops), numt])

    n = mps.num_sites()

    if np.imag(dt) == 0:
        lanczos_fun = lambda x: tf.exp(1j*x)
        lanczos_mult = 1.0
    else:
        lanczos_fun = lambda x: tf.exp(x)
        lanczos_mult = 1.0j

    # Right normalize the state if it's not already
    if ms is None:
        ms = mps.substate(range(n))
    else:
        ms.assign(mps)
    if not ms.is_right_normal(tol):
        ms.left_normalize()
        ms.right_normalize()

    if mps_out[0] is None:
        mps_out[0] = networks.MPS(ms.tensors)
    else:
        mps_out[0].assign(ms)

    if msd is None:
        msd = ms.dagger()
    else:
        msd.assign(ms)
        msd.dagger_equals()

    # Compute the initial energy
    mpo_ms = networks.apply_mpo(mpo, ms, anc_mpo_mps[0])
    eout = (ms.inner(mpo_ms)/ms.inner(ms))[tf.newaxis]
    for exp_idx in range(len(exp_ops)):
        mpo_ms = networks.apply_mpo(exp_ops[exp_idx], ms, anc_mpo_mps[exp_idx+1])
        val = (ms.inner(mpo_ms)/ms.inner(ms))[tf.newaxis]
        if exp_idx == 0:
            it_exp_out = val
        else:
            it_exp_out = tf.stack([it_exp_out, val], axis=0)
        exp_out = it_exp_out[:,tf.newaxis]

    # If we received a final energy, track the sign of the delta
    # so we know when to stop
    de = 0
    if ef is not None:
        de = np.sign(eout[0] - ef)

    # Initialize the R list
    R = [tf.ones([1,1,1], dtype=tf.complex128)]
    ten = tf.tensordot(mpo.tensors[n-1], ms.tensors[n-1], [[2],[2]])
    ten = tf.tensordot(ten, msd.tensors[n-1], [[2],[2]])
    R.insert(0, tf.transpose(ten, perm=[2,0,4,3,1,5]))
    for ii in range(n-3,-1,-1):
        ten = tf.tensordot(R[0], ms.tensors[ii+1], [[0],[1]])
        ten = tf.tensordot(ten, mpo.tensors[ii+1], [[0,6],[1,2]])
        ten = tf.tensordot(ten, msd.tensors[ii+1], [[0,6],[1,2]])
        R.insert(0, tf.transpose(ten, perm = [3,4,5,0,1,2]))

    # Initialize the L list
    L = [tf.ones([1,1,1], dtype=tf.complex128)]
    for ii in range(n-1):
        L.append(None)

    startidx = 0
    idxinc = 1
    endidx = n-1

    itidx = 0
    t = 0
    while itidx < numt-1:
        # Sweep to the right/left updating tensors
        for ii in range(startidx, endidx+idxinc, idxinc):
            nextidx = ii + idxinc
            
            # Get the R tensor
            TR = R[ii]
            if ii < n-1:
                TR = tf.squeeze(TR, axis=[3,4,5])
                #while TR.shape[-1] == 1 and tf.rank(TR) > 3:
                #    TR = tf.squeeze(TR, axis=-1)
            
            # Get the L tensor
            TL = L[ii]
            if ii > 0:
                TL = tf.squeeze(TL, axis=[3,4,5])
                #while TL.shape[-1] == 1 and tf.rank(TL) > 3:
                #    TL = tf.squeeze(TL, axis=-1)
            
            # Evolve according to H
            v = ms.tensors[ii]
            mdims = v.shape
            nv = tf.math.sqrt(tf.reduce_sum(tf.math.conj(v)*v))
            v = v/nv
            num_elms = np.prod(mdims)
            lsteps = min([max([int(num_elms*0.05),2]), num_elms-1, 10])
            v = lanczos_expm_mps(TL*(-lanczos_mult*dt/2.0), TR, [mpo.tensors[ii]], v, lsteps, lanczos_fun)*nv
            
            M2 = tf.reshape(v, mdims)
            
            # Update the tensors
            if idxinc > 0:
                # Left normalize
                M2 = tf.transpose(M2, perm=[0,2,1])
                M2 = tf.reshape(M2, [-1, M2.shape[2]])
                ts, tu, tv = operations.svd_trunc(M2, eps)
                ts = tf.cast(tf.linalg.diag(ts), dtype=M2.dtype)
                new_m = tf.reshape(tu, [mdims[0], mdims[2], mdims[1]])
                new_m = tf.transpose(new_m, perm=[0,2,1])
                
                C = tf.matmul(ts, tf.transpose(tv, conjugate=True))
            else:
                # Right normalize
                M2 = tf.reshape(M2, [M2.shape[0], -1])
                ts, tu, tv = operations.svd_trunc(M2, eps)
                ts = tf.cast(tf.linalg.diag(ts), dtype=M2.dtype)
                new_m = tf.transpose(tv, conjugate=True)
                new_m = tf.reshape(new_m, mdims)
                
                C = tf.matmul(tu, ts)
            
            ms.set_tensor(ii, new_m, False)
            msd.set_tensor(ii, tf.math.conj(new_m), False)
            
            # Update the L/R tensor for this site
            if idxinc > 0:
                if ii == 0:
                    # Initialize the L list
                    ten = tf.tensordot(mpo.tensors[0], ms.tensors[0], [[2],[2]])
                    ten = tf.tensordot(ten, msd.tensors[0], [[2],[2]])
                    L[ii+1] = tf.transpose(ten, perm=[3,1,5,2,0,4])
                elif ii < n-1:
                    # Update the L list
                    ten = tf.tensordot(L[ii], ms.tensors[ii], [[0],[0]])
                    ten = tf.tensordot(ten, mpo.tensors[ii], [[0,6],[0,2]])
                    ten = tf.tensordot(ten, msd.tensors[ii], [[0,6],[0,2]])
                    L[ii+1] = tf.transpose(ten, perm=[3,4,5,0,1,2])
                
                if nextidx <= n-1:
                    TL = L[nextidx]
                    if tf.rank(TL) > 3:
                        TL = tf.squeeze(TL, axis=[3,4,5])
            else:
                if ii == n-1:
                    # Initialize the R list
                    ten = tf.tensordot(mpo.tensors[n-1], ms.tensors[n-1], [[2],[2]])
                    ten = tf.tensordot(ten, msd.tensors[n-1], [[2],[2]])
                    R[ii-1] = tf.transpose(ten, perm=[2,0,4,3,1,5])
                elif ii > 0:
                    # Update the R list
                    ten = tf.tensordot(R[ii], ms.tensors[ii], [[0],[1]])
                    ten = tf.tensordot(ten, mpo.tensors[ii], [[0,6],[1,2]])
                    ten = tf.tensordot(ten, msd.tensors[ii], [[0,6],[1,2]])
                    R[ii-1] = tf.transpose(ten, perm=[3,4,5,0,1,2])
                
                if nextidx >= 0:
                    TR = R[nextidx]
                    if tf.rank(TR) > 3:
                        TR = tf.squeeze(TR, axis=[3,4,5])
            
            if nextidx >= 0 and nextidx <= n-1:
                # Evolve the C tensor backwards in time and contract it into
                # the next two site block
                
                # Evolve according to K
                mdims = tf.shape(C)
                nv = tf.math.sqrt(tf.reduce_sum(tf.math.conj(C)*C))
                C = C/nv
                if mdims[0] is None:
                    num_elms = 1
                else:
                    num_elms = tf.reduce_prod(mdims)
                num_elms = tf.cast(num_elms, tf.float64)
                lsteps = tf.reduce_min([tf.reduce_max([tf.cast(tf.floor(num_elms*0.05), tf.int32),2]), tf.cast(num_elms-1, tf.int32), 10])
                v = lanczos_expm_mps(TL*(lanczos_mult*dt/2.0), TR, [], C, lsteps, lanczos_fun)*nv
                
                C = tf.reshape(v, mdims)
                
                # Compute the next site tensor
                if idxinc > 0:
                    next_m = tf.tensordot(C, ms.tensors[nextidx],[[1],[0]])
                else:
                    next_m = tf.tensordot(ms.tensors[nextidx], C,[[1],[0]])
                    next_m = tf.transpose(next_m, perm=[0, 2, 1])
                
                # Update the next site
                ms.set_tensor(nextidx, next_m, False)
                msd.set_tensor(nextidx, tf.math.conj(next_m), False)
            
            if ms.eager:
                ms.validate()
                msd.validate()

                if np.imag(dt) == 0:
                    assert(abs(ms.inner(ms) - 1.0) < 1e-6)
                    assert(abs(msd.inner(msd) - 1.0) < 1e-6)
        
        # Flip the sweep direction
        if idxinc > 0:
            startidx = n-1
            idxinc = -1
            endidx = 0
        else:
            startidx = 0
            idxinc = 1
            endidx = n-1
            
            # Update time and iteration count
            t = t + dt
            itidx = itidx + 1
            
            # Update output variables
            tvec[itidx] = t
            if mps_out[itidx] is None:
                mps_out[itidx] = networks.MPS(ms.tensors)
            else:
                mps_out[itidx].assign(ms)
            
            mpo_ms = networks.apply_mpo(mpo, ms, anc_mpo_mps[0])
            eout = tf.concat([eout,(ms.inner(mpo_ms)/ms.inner(ms))[tf.newaxis]], axis=0)
            for exp_idx in range(len(exp_ops)):
                mpo_ms = networks.apply_mpo(exp_ops[exp_idx], ms, anc_mpo_mps[exp_idx+1])
                val = (ms.inner(mpo_ms)/ms.inner(ms))[tf.newaxis]
                if exp_idx == 0:
                    it_exp_out = val
                else:
                    it_exp_out = tf.stack([it_exp_out, val], axis=0)
            if len(exp_ops) > 0:
                exp_out = tf.concat([exp_out, it_exp_out[:,tf.newaxis]], axis=1)
            
            if de != 0:
                if de != np.sign(eout[itidx] - ef):
                    eout = eout[itidx+1]
                    tvec = tvec[itidx+1]
                    mps_out = mps_out[itidx+1]
                    break
            
            if debug:
                if itidx % 10 == 0:
                    print(f't = {t}')

    return tvec, eout, exp_out

def tdvp2(mpo, mps, dt, tfinal, eps=0.0, maxrank=0, debug=False, ef=None, exp_ops=[]):
    '''
    An implementation of the time-dependent variational principle algortihm found in
    J. Haegeman, F. Verstraete, et. al "Unifying time evolution and optimization with matrix product states" (2015)
    See in particular equations (5) and (6), as well as the discussion immediately following

    Inputs:
    mpo - The Hamiltonian as an MPO
    mps - The initial MPS
    dt - Time step
    tfinal - The total time over which to evolve the state
    eps - Allowed error for normalization of the MPS
    maxrank - Maximum bond dimension allowed between sites
    debug - Allow verbose output if true
    ef - If provided, evolution will stop once this energy threshold is crossed
    exp_ops - List of operators in MPO form to perform expected value calculations for at each time step

    Returns:
    tvec - Vector of times
    mps_out - List of MPS states at each time
    eout - Expected energy values at each time
    exp_out - [len(exp_ops), num_times] matrix of expected operator values
    '''
    tol = 1e-12

    numt = int(abs(tfinal)/abs(dt) + 1)
    tvec = np.zeros(numt, dtype=np.array(dt).dtype)
    mps_out = [None for _ in range(numt)]
    eout = np.zeros(numt, dtype=np.cdouble)
    exp_out = np.zeros([len(exp_ops), numt], dtype=np.cdouble)

    n = mps.num_sites()

    if np.imag(dt) == 0:
        lanczos_fun = lambda x: np.exp(1j*x)
        lanczos_mult = 1.0
    else:
        lanczos_fun = lambda x: np.exp(x)
        lanczos_mult = 1.0j

    # Right normalize the state if it's not already
    ms = mps.substate(range(n))
    if not ms.is_right_normal(tol):
        ms.left_normalize()
        ms.right_normalize()

    mps_out[0] = networks.MPS(ms.tensors, eager=True)
    msd = ms.dagger()

    # Compute the initial energy
    mpo_ms = networks.apply_mpo(mpo, ms)
    eout[0] = ms.inner(mpo_ms)/ms.inner(ms)
    for exp_idx in range(len(exp_ops)):
        mpo_ms = networks.apply_mpo(exp_ops[exp_idx], ms)
        exp_out[exp_idx, 0] = ms.inner(mpo_ms)/ms.inner(ms)

    # If we received a final energy, track the sign of the delta
    # so we know when to stop
    de = 0
    if ef is not None:
        de = np.sign(eout[0] - ef)

    # Initialize the R list
    R = [tf.ones([1,1,1], dtype=tf.complex128)]
    ten = tf.tensordot(mpo.tensors[n-1], ms.tensors[n-1], [[2],[2]])
    ten = tf.tensordot(ten, msd.tensors[n-1], [[2],[2]])
    R.insert(0, tf.transpose(ten, perm=[2,0,4,3,1,5]))
    for ii in range(n-3,-1,-1):
        ten = tf.tensordot(R[0], ms.tensors[ii+1], [[0],[1]])
        ten = tf.tensordot(ten, mpo.tensors[ii+1], [[0,6],[1,2]])
        ten = tf.tensordot(ten, msd.tensors[ii+1], [[0,6],[1,2]])
        R.insert(0, tf.transpose(ten, perm = [3,4,5,0,1,2]))

    # Initialize the L list
    L = [tf.ones([1,1,1], dtype=tf.complex128)]
    for ii in range(n-1):
        L.append(None)

    startidx = 0
    idxinc = 1
    endidx = n-2

    itidx = 0
    t = 0
    while itidx < numt-1:
        # Sweep to the right/left updating tensors
        for ii in range(startidx, endidx+idxinc, idxinc):
            nextidx = ii + idxinc
            
            # Get the R tensor
            if idxinc > 0:
                ridx = ii + 1
            else:
                ridx = ii
            TR = R[ridx]
            if ridx < n-1:
                while TR.shape[-1] == 1 and tf.rank(TR) > 3:
                    TR = tf.squeeze(TR, axis=-1)
            
            # Get the L tensor
            if idxinc > 0:
                lidx = ii
            else:
                lidx = ii - 1
            TL = L[lidx]
            if lidx > 0:
                while TL.shape[-1] == 1 and tf.rank(TL) > 3:
                    TL = tf.squeeze(TL, axis=-1)
            
            # Contract the current tensors
            v = tf.tensordot(ms.tensors[ridx-1], ms.tensors[ridx],[[1],[0]])
            v = tf.transpose(v, perm=[0,2,1,3])
            vdims = v.shape

            # Evolve according to H
            nv = tf.math.sqrt(tf.reduce_sum(tf.math.conj(v)*v))
            v = v/nv
            num_elms = np.prod(vdims)
            lsteps = min([max([int(num_elms*0.05),2]), num_elms-1, 10])
            v = lanczos_expm_mps(TL*(-lanczos_mult*dt/2.0), TR, [mpo.tensors[ridx-1], mpo.tensors[ridx]], v, lsteps, lanczos_fun)*nv
            
            M2 = tf.reshape(v, vdims)
            M2 = tf.transpose(M2, perm=[0,2,1,3])
            M2 = tf.reshape(M2, [M2.shape[0]*M2.shape[1],-1])
            ts, tu, tv = operations.svd_trunc(M2, eps, maxrank)
            ts = tf.cast(tf.linalg.diag(ts), dtype=M2.dtype)
            
            # Update the tensors
            if idxinc > 0:
                # Left normalize
                new_m = tf.reshape(tu, [vdims[0], vdims[2], -1])
                new_m = tf.transpose(new_m, perm=[0,2,1])
                
                C = tf.matmul(ts, tf.transpose(tv, conjugate=True))
                C = tf.reshape(C, [-1,vdims[1],vdims[3]])
            else:
                # Right normalize
                new_m = tf.transpose(tv, conjugate=True)
                new_m = tf.reshape(new_m, [-1,vdims[1],vdims[3]])
                
                C = tf.matmul(tu, ts)
                C = tf.reshape(C, [vdims[0],vdims[2],-1])
                C = tf.transpose(C, perm=[0,2,1])
            
            ms.set_tensor(ii, new_m, False)
            msd.set_tensor(ii, tf.math.conj(new_m), False)
            
            # Update the L/R tensor for this site
            if idxinc > 0:
                if ii == 0:
                    # Initialize the L list
                    ten = tf.tensordot(mpo.tensors[0], ms.tensors[0], [[2],[2]])
                    ten = tf.tensordot(ten, msd.tensors[0], [[2],[2]])
                    L[ii+1] = tf.transpose(ten, perm=[3,1,5,2,0,4])
                elif ii < n-1:
                    # Update the L list
                    ten = tf.tensordot(L[ii], ms.tensors[ii], [[0],[0]])
                    ten = tf.tensordot(ten, mpo.tensors[ii], [[0,6],[0,2]])
                    ten = tf.tensordot(ten, msd.tensors[ii], [[0,6],[0,2]])
                    L[ii+1] = tf.transpose(ten, perm=[3,4,5,0,1,2])
                
                if nextidx <= n-1:
                    TL = L[nextidx]
                    while TL.shape[-1] == 1 and tf.rank(TL) > 3:
                        TL = tf.squeeze(TL, axis=-1)
            else:
                if ii == n-1:
                    # Initialize the R list
                    ten = tf.tensordot(mpo.tensors[n-1], ms.tensors[n-1], [[2],[2]])
                    ten = tf.tensordot(ten, msd.tensors[n-1], [[2],[2]])
                    R[ii-1] = tf.transpose(ten, perm=[2,0,4,3,1,5])
                elif ii > 0:
                    # Update the R list
                    ten = tf.tensordot(R[ii], ms.tensors[ii], [[0],[1]])
                    ten = tf.tensordot(ten, mpo.tensors[ii], [[0,6],[1,2]])
                    ten = tf.tensordot(ten, msd.tensors[ii], [[0,6],[1,2]])
                    R[ii-1] = tf.transpose(ten, perm=[3,4,5,0,1,2])
                
                if nextidx >= 0:
                    TR = R[nextidx]
                    while TR.shape[-1] == 1 and tf.rank(TR) > 3:
                        TR = tf.squeeze(TR, axis=-1)
            
            if nextidx >= 1 and nextidx <= n-2:
                # Evolve the second site backwards in time and contract it into
                # the next two site block
                
                # Evolve according to H
                mdims = C.shape
                nv = tf.math.sqrt(tf.reduce_sum(tf.math.conj(C)*C))
                C = C/nv
                num_elms = np.prod(mdims)
                lsteps = min([max([int(num_elms*0.05),2]), num_elms-1, 10])
                v = lanczos_expm_mps(TL*(lanczos_mult*dt/2.0), TR, [mpo.tensors[nextidx]], C, lsteps, lanczos_fun)*nv
                
                next_m = tf.reshape(v, mdims)
                
            else:
                # The next site is an end site, so normalize the C tensor as 
                # appropriate and put it in the MPS. For an end site, this 
                # throws away a scalar, effectively normalizing the MPS
                
                cdims = C.shape
                if idxinc > 0:
                    # Left normalize
                    C = tf.transpose(C, perm=[0,2,1])
                    C = tf.reshape(C, [C.shape[0]*C.shape[1],-1])
                    _, tu, _ = tf.linalg.svd(C)
                    next_m = tf.reshape(tu, [cdims[0],cdims[2],-1])
                    next_m = tf.transpose(next_m, perm=[0,2,1])
                else:
                    # Right normalize
                    C = tf.reshape(C, [C.shape[0], -1])
                    _, _, tv = tf.linalg.svd(C)
                    next_m = tf.reshape(tf.transpose(tv, conjugate=True), [-1,cdims[1], cdims[2]])
                
            # Update the next site
            ms.set_tensor(nextidx, next_m, False)
            msd.set_tensor(nextidx, tf.math.conj(next_m), False)
            
            ms.validate()
            msd.validate()

            if np.imag(dt) == 0:
                assert(abs(ms.inner(ms) - 1.0) < 1e-6)
                assert(abs(msd.inner(msd) - 1.0) < 1e-6)
        
        # Flip the sweep direction
        if idxinc > 0:
            startidx = n-1
            idxinc = -1
            endidx = 1
        else:
            startidx = 0
            idxinc = 1
            endidx = n-2
            
            # Update time and iteration count
            t = t + dt
            itidx = itidx + 1
            
            # Update output variables
            tvec[itidx] = t
            mps_out[itidx] = networks.MPS(ms.tensors, eager=True)
            
            mpo_ms = networks.apply_mpo(mpo, ms)
            eout[itidx] = ms.inner(mpo_ms)/ms.inner(ms)
            for exp_idx in range(len(exp_ops)):
                mpo_ms = networks.apply_mpo(exp_ops[exp_idx], ms)
                exp_out[exp_idx, itidx] = ms.inner(mpo_ms)/ms.inner(ms)
            
            if de != 0:
                if de != np.sign(eout[itidx] - ef):
                    eout = eout[itidx+1]
                    tvec = tvec[itidx+1]
                    mps_out = mps_out[itidx+1]
                    break
            
            if debug:
                if itidx % 10 == 0:
                    print(f't = {t}, max rank = {np.max(ms.rank())}')

    return tvec, mps_out, eout, exp_out
