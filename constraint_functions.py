#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:50:14 2018

@author: Daniel Puzzuoli

Description:
    This file contains various functions that are meant to implement
    constraints on control amplitudes. The base function is
    mono_constraint_func, standing for monomial constraint function, which
    is zero in some region, and rises rapidly once you leave the region,
    where the rise is given as a power of how close you are to the boundary
    (see the function description for details).
    
    The other functions are built on mono_constraint_func, currently 
    mono_power_constraint, which computes a penalty on control power for a
    list of amplitudes, and mono_smoothness_constraint, which computes a
    penalty on individual control amplitudes based on how quickly they are
    changing.
    
    Currently, these functions compute the penalty, as well as the jacobian,
    and Hessian computation still needs to be implemented.
    

To do:
    - implement constraints that are functions of multiple control amplitudes,
    e.g. x^2 + y^2 <= ub - epsilon
    - implement Hessian computation
"""
from numpy import empty,array,ndarray,zeros

def mono_objective(x, power_lb, power_ub, power_eps, change_b = None, change_eps = None,n=2, deriv = 0):
    
    # first, compute f(x)
    val = mono_power_constraint(x, power_lb, power_ub, n, power_eps, 0)
    
    if change_b is not None:
        val = val + mono_smoothness_constraint(x, change_b, n, change_eps, 0)
    
    # if deriv == 0, return the function value
    if deriv == 0:
        return val
    
    
    # next, compute f'(x)
    jac = mono_power_constraint(x, power_lb, power_ub, n, power_eps, 1)
    if change_b is not None:
        jac = jac + mono_smoothness_constraint(x, change_b, n, change_eps, 1)
    
    #if deriv == 1, return the function value and derivative
    if deriv == 1:
        return val, jac
    
    # compute hessian
    hess = mono_power_constraint(x, power_lb, power_ub, n, power_eps, 2)
    if change_b is not None:
        hess = hess + mono_smoothness_constraint(x, change_b, n, change_eps, 2)
    
    if deriv == 2:
        return val, jac, hess


def mono_smoothness_constraint(x,b, n, eps, deriv=0):
    """
    Given a 1d array b representing bounds on the rate of change of each
    control amplitude, calculates a penalty for rate of change.
    
    Specifically, for a single control amplitude, is meant to penalize controls
    not satisfying |x[i] - x[i+1]| <= b, where the penalty is again computed
    via mono_constraint_func.
    
    Note that this function actually calls mono_power_constraint, as smoothness
    constraints may be viewed as "power constraints" on the differences of the
    amplitudes at every time step.
    
    Parameters
    ----------
    x : list of lists or 1d or 2d numpy.array
        the values to be penalized, will represent control amplitudes
    b : int, float, list, or numpy.array
        the bound on rate of change for each control amplitude
    n : int
        the power of the monomial to be used
    eps : float
        the distance to the boundaries at which the constraints kick in
    deriv : int
        which derivative to compute (with deriv ==0 corresponding to the base
        function itself)     

    Returns
    -------
    f(x), jac(f)(x) or hess(f)(x).
    Note: as written hess(f) can be simplified
    """
    
    b = to_1d_array(b)
    x = to_2d_array(x)
    
    # if len(x) == 1, then smoothness constraints have no meaning
    if len(x) == 1:
        return 0
    
    if deriv == 0:
        # for the 0 derivative case, can just call mono-power constraints
        # on the step by step differences
        return mono_power_constraint(x[:-1] - x[1:], -b, b, n, eps, 0)
    if deriv == 1:
        # evaluate the constraint derivatives on each pair of differences
        term_derivs = mono_power_constraint(x[:-1] - x[1:], -b, b, n, eps, 1)
        
        jac = empty(x.shape)
        
        # the derivatives with respect to the first and last time steps
        # are only influenced by the first and last entries of term_derivs
        jac[0] = term_derivs[0]
        jac[-1] = -term_derivs[-1]
        # populate the rest of jac for the control amplitudes for interior
        # time steps. We get the derivative by simply subtracting the derivatives
        # e.g. suppose for some control parameter x we have x1,x2,x3 at time steps
        # 1, 2, 3. In the full constraint x2 contributes to f(x1-x2) + f(x2-x1). 
        # and hence (d/dx2)(f(x1-x2) + f(x2-x3)) = -f'(x1-x2)+f'(x2-x3)
        jac[1:-1] = -term_derivs[0:-1] + term_derivs[1:]
        
        return jac
    if deriv == 2:
        N,dc = x.shape
        
        
        #initialize to zeros as almost all will be zero
        hess = zeros((N,dc,N,dc))
        
        # currently, smoothness is assumed to be an amplitude by amplitude
        # property
        for ctrl_i in range(dc):
            
            #first, the second derivatives at the same time step, starting with
            # the beginning and endpoints
            hess[0,ctrl_i,0,ctrl_i] = mono_constraint_func(x[0,ctrl_i]-x[1,ctrl_i], -b[ctrl_i],b[ctrl_i],n,eps,2)
            hess[-1,ctrl_i,-1,ctrl_i] = mono_constraint_func(x[-2,ctrl_i]-x[-1,ctrl_i], -b[ctrl_i],b[ctrl_i],n,eps,2)
            
            for k in range(1,N-1):
                hess[k,ctrl_i,k,ctrl_i] = mono_constraint_func(x[k-1,ctrl_i]-x[k,ctrl_i], -b[ctrl_i],b[ctrl_i],n,eps,2)+ mono_constraint_func(x[k,ctrl_i]-x[k+1,ctrl_i], -b[ctrl_i],b[ctrl_i],n,eps,2)
                
            # next, second derivatives at adjacent time-steps
            
            for k in range(N-1):
                hess[k,ctrl_i,k+1,ctrl_i] = -mono_constraint_func(x[k,ctrl_i]-x[k+1,ctrl_i], -b[ctrl_i],b[ctrl_i],n,eps,2)
                hess[k+1,ctrl_i,k,ctrl_i] = hess[k,ctrl_i,k+1,ctrl_i]
            
        return hess
        
    
def mono_power_constraint(x,lb, ub, n, eps, deriv = 0):
    """
    This is essentially an ``array'' version of mono_constraint_func.
    The lower bounds lb and upper bounds ub are given as a 1d array, 
    with each entry giving the upper and lower bound on each control amplitude.
    
    x is now representing a list of control amplitudes, and
    mono_constraint_func is applied on all control amplitudes at each time
    step, with the results summed together
    
    Parameters
    ----------
    x : list of lists or 1d or 2d numpy.array
        the values to be penalized, will represent control amplitudes
    lb : int, float, list, or numpy.array
        the lower bounds for each entry
    ub : float
        the upper bounds for each entry
    n : int
        the power of the monomial to be used
    eps : float
        the distance to the boundaries at which the constraints kick in
    deriv : int
        which derivative to compute (with deriv ==0 corresponding to the base
        function itself)     

    Returns
    -------
    f(x) or f'(x) ( f''(x) not supported )
    """
    
    # ensures all variables are of the desired type
    ub = to_1d_array(ub)
    lb = to_1d_array(lb)
    x = to_2d_array(x)
    
    
    if deriv == 0:
        # compute the constraint function
        
        # loop through every entry of x and compute the penalty
        val = 0
        for ctrl_i in range(len(lb)):
            for tstep in range(len(x)):
               val = val + mono_constraint_func(x[tstep,ctrl_i], lb[ctrl_i], ub[ctrl_i], n, eps, 0)
        
        return val
    
    elif deriv == 1:
        # compute the constraint function jacobian
        
        jac = empty(x.shape, dtype = float)
        for ctrl_i in range(len(lb)):
            for tstep in range(len(x)):
                jac[tstep,ctrl_i] = mono_constraint_func(x[tstep,ctrl_i], lb[ctrl_i], ub[ctrl_i], n, eps, 1)
        
        return jac
    
    elif deriv == 2:
        
        N,dc = x.shape
        
        # Initialize the hessian to zeros as the off diagonals will all be 0,
        # as this function has no cross terms
        hess = zeros((N,dc,N,dc))
        
        for ctrl_i in range(dc):
            for k in range(N):
                hess[k,ctrl_i,k,ctrl_i] = mono_constraint_func(x[k,ctrl_i], lb[ctrl_i], ub[ctrl_i], n, eps, 2)
        
        return hess

def mono_constraint_func(x,lb, ub, n, eps, deriv = 0):
    """
    Given a lower bound, upper bound, a power n, epsilon, behaves as the function
    
    f(x) = 0 for lb + eps <= x <= ub - eps
           ((x - (ub - eps))/eps)**n for x > ub - eps
           (((lb + eps)-x)/eps)**n for x < lb + eps
    
    which is viewed as a penalty for x approaching the boundaries lb, ub, where
    eps controls where the penalty kicks in. The function f is defined so that
    f(lb) = f(ub) = 1. 
    
    if deriv == 0, returns f(x)
    if deriv == 1, returns f'(x)
    if deriv == 2, returns f''(x)
    
    Parameters
    ----------
    x : float
    lb : float
    ub : float
    n : int
    eps : float
    deriv : int    

    Returns
    -------
    see description
    """
    
    if deriv == 0:
        if lb +eps <= x <= ub - eps:
            return 0
        elif x > ub - eps:
            return ((x - (ub - eps))/eps)**n
        elif x < lb + eps:
            return (((lb + eps) - x)/eps)**n
    elif deriv == 1:
        if lb +eps <= x <= ub - eps:
            return 0
        elif x > ub - eps:
            return n*(((x - (ub - eps))/eps)**(n-1))/eps
        elif x < lb + eps:
            return n*((((lb + eps) - x)/eps)**(n-1))/(-eps)
    elif deriv == 2:
        if lb +eps <= x <= ub - eps:
            return 0
        elif x > ub - eps:
            return n*(n-1)*(((x - (ub - eps))/eps)**(n-2))/(eps**2)
        elif x < lb + eps:
            return n*(n-1)*((((lb + eps) - x)/eps)**(n-2))/(eps**2)
        
def to_2d_array(x):
    """
    converts ints, floats, lists, and 1d numpy arrays to 2d numpy arrays
    """
    if (type(x) == int) | (type(x) == float):
        return array([[x]])
    if type(x) == list:
        x = array(x)
    
    if type(x) == ndarray:
        if len(x.shape)==1:
            return array([x])
        
    return x

        
def to_1d_array(x):
    """
    converts ints, floats, and lists into a 1d numpy array
    """
    if (type(x) == int) | (type(x) == float):
        return array([x])
    if type(x) == list:
        return array(x)
    
    return x