import tensorflow as tf
import numpy as np
import networks
import operations

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
