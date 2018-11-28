#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:50:24 2018

@author: Daniel Puzzuoli

Description:
    The main function in this file is evolve_system, which given a control
    system, amplitudes, and times, computes the final propagator, as well as
    derivatives w.r.t. control amplitudes to a desired level. Other functions
    are helpers.

Notes:
    The format of the function calls may change once I introduce transfer
    functions, as I may absorb time step lengths into transfer functions.

To do:
    - Implement Hessian computation
    - Eventually, I want to be able to have objectives that depend on
    intermediate propagators, not just the one at the final time. When this
    is implemented it may require a complete overhaul of the functionality here
"""

from utb_matrices import decoupling_gen
from numpy import array,empty,identity,matmul
from vexpm import vexpm

def evolve_system(system, amps, dt, deriv=0):
    """
    Given a control system, a list of amplitudes at various time steps, and
    the size of each time step dt, computes the final propagator
    for the system. Depending on the value of deriv, will also compute
    derivatives with respect to control amplitudes up to a given order.
    
    Parameters
    ----------
    system : control_system object
    amps : numpy.array
        list of control amplitudes for each time step
    dt : float
        length of the time steps
    deriv : int
        level of derivatives to be computed. Currently supports:
            - deriv = 0: no derivative computed
            - deriv = 1: first derivative

    Returns
    -------
    if deriv == 0
        numpy.array Ufinal
        the final propagator
    if deriv == 1
        numpy.array Ufinal and numpy.array final_derivs
        First is the final propagator, second is an array with shape 
        (N = number of time steps ,c = number of control amplitudes ,sys_dim,sys_dim),
        with final_derivs[i,j] = derivative of Ufinal with respect to the j^{th}
        control amplitude at time step i
    if deriv == 2
        not yet supported
    """
    
    # Compute the final propagator
    if deriv == 0:
        
        props = evolve_steps(system, amps, dt)
        final = props[0]
        for k in range(1,len(props)):
            final = final@props[k]
        
        return final
    
    # Compute the first derivative and final propagator
    elif deriv == 1:
        
        # set a few local variables for convenience
        N = len(amps)
        ds = system.system_dim
        dc = system.control_dim
        
        # we get the full generators for each time step, and also multiply each
        # generator by the corresponding time
        full_generators = system.generator_list(amps)*dt
        
        
        # initialize the array of derivatives. Note that the first index is
        # control amplitude, and the second is time step, which is the opposite
        # of how things are usually specified in the code. These indices will
        # be transposed at the end
        full_derivs = empty((dc, N, ds, ds), dtype = complex)
        # Also initialize variable names for the forward and backward props.
        # These will be computed simultaneously with first derivatives.
        forward = None
        backward = None
        for ctrl_i in range(dc):
            
            # first, construct the generators for computing derivatives
            deriv_gen = empty((N, 2*ds,2*ds), dtype = complex)
            for k in range(N):
                deriv_gen[k] = decoupling_gen(full_generators[k], [system.control_generators[ctrl_i]*dt])
            
            #exponentiate to compute derivatives
            deriv_exp = vexpm(deriv_gen)
            
            # if this is the first control amplitude, extract the system
            # propagators and compute forward and backward propagators.
            # The system propagators are in the top left block of deriv_exp
            if ctrl_i == 0:
                forward,backward = forward_backward_from_steps(deriv_exp[:,0:ds,0:ds])
            
            # extract derivatives of single propagators with respect to the 
            # current control amplitude
            single_prop_derivs = deriv_exp[:,0:ds,ds:2*ds]
            
            
            # compute the derivatives of the full propagator
            full_derivs[ctrl_i] = matmul(matmul(forward[0:N],single_prop_derivs),backward)
        
        # return the final propagator, as well as the derivatives of the final
        # propagator w.r.t. control amplitudes. 
        return forward[-1],full_derivs.transpose([1,0,2,3])
                    


def forward_backward_from_steps(prop_steps):
    """
    Computes the forward backward propagators given a list of propagators for
    each time step
    
    Parameters
    ----------
    prop_steps : numpy.array
    
    assumed to be of shape (N,d,d)

    Returns
    -------
    forward,backward - each a numpy array
    
    For input [U1, U2, ..., U(n-1), Un],
        forward = [id, U1, U1@U2, ..., U1@...@Un] (length N + 1)
        backward = [U2@...@Un, U3@...@Un, ..., Un, id] (length N)
    
    This choice of this particular output format is so that:
        - forward[-1] is the final propagator
        - forward[i+1]@backward[i] = forward[-1] for all 0 <= i < N
        - For di a derivative of the i^th propagator (with respect to some
          parameter), the corresponding derivative of the final propagator
          is: 
                 forward[i]@di@backward[i]
    """
    
    # set some local variables for convenience
    N = len(prop_steps)
    d = len(prop_steps[0])
    
    # handle the case N==1 immediately
    if N==1:
        forward = array([identity(d), prop_steps[0]])
        backward = array([identity(d)])
        return forward,backward
    
    # compute the forward propagators
    forward = empty( (N+1, d, d) , dtype = complex )
    
    # initialize the first forward propagator to the identity
    forward[0] = identity(d)
    forward[1] = prop_steps[0]
    for k in range(2, N+1):
        forward[k] = forward[k-1]@prop_steps[k-1]
        
    
    # compute the backward propagators
    backward = empty( (N,d,d) , dtype = complex )
    # initialize the last backward propagator to be the identity, and the
    # second last to the last prop_step
    backward[-1] = identity(d)
    backward[-2] = prop_steps[-1]
    for k in range(3, N+1):
        backward[-k] = prop_steps[-k+1]@backward[-k+1]
    
    return forward, backward
        

def evolve_steps(system, amps, dt):
    """
    Computes a list of single time step propagators for a control system from
    a list of amplitudes and dt, the length of eah time step
    
    Parameters
    ----------
    system : control_system object
    amps : numpy.array
        list of control amplitudes for each time step
    dt : float
        length of the time steps

    Returns
    -------
    numpy.array with shape (len(times), sys_dim, sys_dim)
    
    ith entry is the propagator for the i^th time step
    """
    
    # get all the generators and multiply by time
    generators = system.generator_list(amps)*dt
    
    
    # perform exponentiation on the array of generators
    return vexpm(generators)