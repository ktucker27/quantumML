import numpy as np
import math

def pow_2_exp(a, b, num_terms):
    '''
    Taken from 
    T. Bochud, D. Challet, "Optimal approximations of power-laws with exponentials" (2006)
    Returns values alpha, beta such that
           x^-a ~ sum_i=0^n alpha[i]*beta[i]**x
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
