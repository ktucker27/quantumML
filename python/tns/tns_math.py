import numpy as np
import math

from scipy import optimize

def pow_2_exp(a, b, num_terms):
    '''
    Taken from 
    T. Bochud, D. Challet, "Optimal approximations of power-laws with exponentials" (2006)
    Returns values alpha, beta such that
           x^-a ~ sum_i=0^(num_terms-1) alpha[i]*beta[i]**x
    The value of b gives a sense of scale in x over which the approximation will be most accurate.
    It should be roughly in the middle of the desired range of accuracy
    '''

    # (NB: The MATLAB counterpart to this function defines n to be one greater, i.e. it
    #      returns nx1 vectors, whereas this function returns vectors of length n+1 to be
    #      consistent with the paper)
    n = num_terms-1

    c = np.ones(n+1)
    for kk in range(1,n+1):
        sumval = 0
        for ii in range(kk):
            sumval = sumval + c[n-ii]*b**(-a*(kk-ii))*math.exp(a*(1 - b**(ii-kk)))
        c[n - kk] = 1 - sumval

    alpha = np.zeros(c.shape)
    beta = np.zeros(c.shape)
    for ii in range(n+1):
        alpha[ii] = c[ii]*(math.exp(1)/b**ii)**a
        beta[ii] = math.exp(-(a*b**(-ii)))

    return alpha, beta, c

def exp_loss(alpha_beta, rmult, rpow, rcutoff, n):
    rvec = np.arange(1.0,rcutoff,1.0)

    l = 0.0
    for r in rvec:
        sumval = 0
        for ii in range(n):
            sumval = sumval + alpha_beta[ii]*alpha_beta[n+ii]**(r)
        l = l + (sumval - rmult/r**rpow)**2

    return l

def pow_2_exp_refine(a, b, num_terms, rmult, rcutoff):
    '''
    Returns values alpha, beta such that
           rmult*r^-a ~ sum_i=0^(num_terms-1) alpha[i]*beta[i]**r
    using pow_2_exp and iterative refinement.

    The value of b gives a sense of scale in x over which the approximation will be most accurate.
    It should be roughly in the middle of the desired range of accuracy.

    The value of rcutoff is used to define the full range overwhich the loss function will be evaluated
    '''

    alpha, beta, _ = pow_2_exp(a, b, num_terms)
    ab0 = np.concatenate([alpha,beta])
    fun = lambda alpha_beta : exp_loss(alpha_beta, rmult, a, rcutoff, num_terms)
    bounds = [[-np.Inf,np.Inf]]*alpha.shape[0] + [[0,np.Inf]]*beta.shape[0]
    result = optimize.minimize(fun, x0=ab0, method='L-BFGS-B', bounds=bounds, options={'maxiter':10000})
    ab = result.x
    alpha = ab[:num_terms]
    beta = ab[num_terms:]

    return alpha, beta, result.success