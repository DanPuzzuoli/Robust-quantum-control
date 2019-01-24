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
from numpy import array,empty,identity
from vexpm import vexpm
from matmult import matmult

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
        See Notes.
        
    Notes on implementing deriv == 2
    --------------------------------
    Update 2: Should double check that no redundancies are happening in second
    derivative computation. Specifically, no unnecessary matrix multiplications,
    and also make sure we are only computing the minimal number of Hessian
    elements. I think this is the case already but it should be gone through
    more carefully when time is available.
    
    Update 1: I believe deriv == 2 is working. I've run some tests using the
    prebuilt xy system by plugging some stuff in mathematica for the N=1,2,3
    cases and seeing if the output is the same. So far it has passed every test.
    Ultimately it would be good to write some proper test cases; mathematica
    is useful for quick testing as it is easy to write up expressions for taking
    analytic derivatives. One option is to then transfer these expressions to
    Python. Another option is to just compute the derivatives using finite
    differences and compare the results. This latter option is better but
    I will leave it for another day. That the propagator and its jacobian
    are computed correctly has been verified on the assumption that the
    deriv == 1 case is correct.
    
    Update 0: deriv == 2 is almost done, but need to test/confirm everything. Current
    formatting is the Hessian is given as an (N,dc,N,dc,ds,ds) array, where the
    derivative of the final propagator w.r.t. ctrl_i at time step k, and ctrl_j
    at time step l is in (k, ctrl_i, l, ctrl_j). The full array however is not
    populated, as the entry (k, ctrl_i, l, ctrl_j) and (l, ctrl_j, k, ctrl_i)
    will actually be the same due to the derivatives commuting. So, currently,
    it is populated so that the entry which is earlier in lexicographic ordering
    is populated, and hte other isn;t. That is, we say (k,ctrl_i)<= (l, ctrl_j)
    if either k < l, or k = l and ctrl_i <= j. The rest of the code can either
    be designed to alway take this into account, or we could define an object
    (that potentially stores the prop, jacobian, and hessian, or just the
    hessian) which can be called without respecting lexicographic ordering,
    but automatically does the proper conversion.
    
    General notes
    -------------
    At some point, consolodate notation in the various files, 
    e.g. use U for final propagator, Ujac, Uhess, Usteps, Uintervals, etc...
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
            full_derivs[ctrl_i] = matmult(matmult(forward[0:N],single_prop_derivs),backward)
        
        # return the final propagator, as well as the derivatives of the final
        # propagator w.r.t. control amplitudes. 
        return forward[-1],full_derivs.transpose([1,0,2,3])
    
    # compute the final propagator, first, and second derivatives with
    # respect to control amplitudes
    
    if deriv == 2:
        
        # set a few local variables for convenience
        N = len(amps)
        ds = system.system_dim
        dc = system.control_dim
        
        # get the full generators for each time step, and also multiply each
        # generator by the corresponding time
        full_generators = system.generator_list(amps)*dt
        
        
        # initialize variable for array of propagator intervals. 
        # The whole array will be assigned at once in the first iteration
        # of the loop
        prop_intervals = None
        # initialize an array for first derivatives of propagator steps
        prop_steps_d1 = empty((dc, N, ds, ds), dtype = complex)
        # initialize an array for matrices that the second derivatives
        # will be drawn from
        prop_steps_d2_part = empty((dc,dc,N,ds,ds), dtype = complex)
        
        # first populate the above arrays
        for ctrl_i in range(dc):
            for ctrl_j in range(dc):
                
                # we compute the derivatives using block matrix methods
                deriv_gen = deriv_gen = empty((N, 3*ds,3*ds), dtype = complex)
                
                for k in range(N):
                    deriv_gen[k] = decoupling_gen(full_generators[k], [system.control_generators[ctrl_i]*dt, system.control_generators[ctrl_j]*dt])
                
                deriv_exp = vexpm(deriv_gen)
                
                # if this is the first system, extract the propagator steps.
                # and compute intervals from top right block
                if (ctrl_i == 0) and (ctrl_j == 0):
                    prop_intervals = propagator_intervals_from_steps(deriv_exp[:,0:ds, 0:ds])
                    
                
                # if this is the first ctrl_j for the corresponding ctrl_i,
                # populate the first derivative for ctrl_i
                if (ctrl_j == 0):
                    prop_steps_d1[ctrl_i] = deriv_exp[:,0:ds,ds:2*ds]
                
                # populate the second derivative parts
                prop_steps_d2_part[ctrl_i,ctrl_j] = deriv_exp[:,0:ds,2*ds:3*ds]
        
        ################################
        # Handle case N==1
        ################################
        
        # The code beyond this block works for N >= 2, and is mainly constructed
        # around the the issues of the multiplications required in the N>=2 case.
        # So, we do the N==1 case first; in later iterations we can probably
        # write both cases at once
        
        # In the N==1 case: 
        #        -the final propagator is just prop_intervals[0,0]
        #        -the first derivatives are just the prop_steps_d1 transposed
        #        -the second derivatives are just the second derivatives of the
        #         single propagator
        
        if N==1:
            full_prop_d2 = empty((dc,N,dc,N,ds,ds), dtype = complex)
            
            for ctrl_i in range(dc):
                for ctrl_j in range(ctrl_i,dc):
                    
                    full_prop_d2[ctrl_i,0,ctrl_j,0] = prop_steps_d2_part[ctrl_i,ctrl_j,0] + prop_steps_d2_part[ctrl_j,ctrl_i,0]
            
            return prop_intervals[0,0],prop_steps_d1.transpose([1,0,2,3]),full_prop_d2.transpose([1,0,3,2,4,5])
        
        #################################
        # Compute first derivatives
        #################################
        
        
        # when computing the first derivatives, the "forward" 
        # drivatives are computed, which are also needed when computing
        # second derivatives, so we store them separately to save on
        # multiplications
        forward_d1 = empty((dc,N-1,ds,ds), dtype = complex)
        
        # initialize full derivative variable
        full_prop_d1 = empty((dc, N, ds, ds), dtype = complex)
        
        
        for ctrl_i in range(dc):
            # forward_d1[ctrl_i,0] is just the derivative of the first
            # propagator
            forward_d1[ctrl_i,0] = prop_steps_d1[ctrl_i,0]
            
            # populate the first and last
            full_prop_d1[ctrl_i,0] = prop_steps_d1[ctrl_i,0]@prop_intervals[1,N-1]
            full_prop_d1[ctrl_i,N-1] = prop_intervals[0,N-2]@prop_steps_d1[ctrl_i,N-1]
            
            #populate the interior values
            for k in range(1,N-1):
                forward_d1[ctrl_i,k] = prop_intervals[0,k-1]@prop_steps_d1[ctrl_i,k]
                full_prop_d1[ctrl_i,k] = forward_d1[ctrl_i,k]@prop_intervals[k+1,N-1]
            

        #################################
        # Compute second derivatives
        #################################
        full_prop_d2 = empty((dc,N,dc,N,ds,ds), dtype = complex)
        
        # first populate terms for second derivatives at same time step
        # this uses the same multiplication steps as first derivs, so could
        # be factored out later
        for ctrl_i in range(dc):
            for ctrl_j in range(ctrl_i,dc):
                
                # populate array of shape (N,ds,ds), where kth entry is
                # the second order deriv of prop step k wrt to controls i and j
                single_prop_d2= prop_steps_d2_part[ctrl_i,ctrl_j] + prop_steps_d2_part[ctrl_j,ctrl_i]
                
                # populate the first and last
                full_prop_d2[ctrl_i,0,ctrl_j,0] = single_prop_d2[0]@prop_intervals[1,N-1]
                full_prop_d2[ctrl_i,N-1,ctrl_j,N-1] = prop_intervals[0,N-2]@single_prop_d2[N-1]
                
                #populate the interior values
                for k in range(1,N-1):
                    full_prop_d2[ctrl_i,k,ctrl_j,k] = prop_intervals[0,k-1]@single_prop_d2[k]@prop_intervals[k+1,N-1]
        
        # populate terms not corresponding to derivatives at the same time
        # step
        
        # Need to spend more time thinking of the most efficient way to do this,
        # but the current plan is to create forward and backward props for each
        # control, then to compute the second derivs we sandwich these with
        # the intermediate interval. Need to then take care of edge cases
        # Forward derivs are already computed in first derivative computation,
        # so compute the backward ones
        backward_d1 = empty((dc,N-1,ds,ds), dtype = complex)
        
        for ctrl_i in range(dc):
            for k in range(N-2):
                backward_d1[ctrl_i,k] = prop_steps_d1[ctrl_i,k+1]@prop_intervals[k+2,N-1]
            
            backward_d1[ctrl_i,-1] = prop_steps_d1[ctrl_i,-1]
        
        # loop over every choice for the first time step
        for k in range(N-1):
            # loop overy every choice for the ctrl parameter at the first
            # time step
            for ctrl_i in range(dc):
                
                # first populate the second derivative with the second
                # time step being immediately after
                for ctrl_j in range(dc):
                    full_prop_d2[ctrl_i, k, ctrl_j, k+1] = forward_d1[ctrl_i,k]@backward_d1[ctrl_j,k]
                
                # next, populate the second derivatives for the second time
                # step l satisfying l > k + 1
                for l in range(k+2,N):
                    # multiply the forward derivative with the propagator
                    # interval between the time steps
                    forward_sandwich = forward_d1[ctrl_i,k]@prop_intervals[k+1,l-1]
                    
                    for ctrl_j in range(dc):
                        full_prop_d2[ctrl_i, k, ctrl_j,l] = forward_sandwich@backward_d1[ctrl_j,l-1]
            
            
                
        return prop_intervals[0,-1],full_prop_d1.transpose([1,0,2,3]),full_prop_d2.transpose([1,0,3,2,4,5])
            
        
          
def propagator_intervals_from_steps(prop_steps):
    """
    Given propagator steps, returns an array whose (i,j)th entry is the
    propagator over time step [i, j+1] for i <= j. For i > j, is empty, where
    we are using the convention that the propagator over the interval [i, i+1]
    is U(i+1), with Python indexing starting at 0, but time-step indexing 
    starting at 1
    
    Parameters
    ----------
    prop_steps : numpy.array
    
    assumed to be of shape (N,d,d)

    Returns
    -------
    prop_intervals - numpy array of shape (N,N,d,d)
    
    on input [U1, U2, U3, U4], outputs
    
    [ [U1, U1@U2, U1@U2@U3, U1@U2@U3@U4],
      [0,   U2,     U2@U3,    U2@U3@U4 ],
      [0,    0,       U3,        U3@U4 ],
      [0,    0,        0,          U4  ]
    ]
    where the 0s are not actually 0, but the result from the initialization
    of an empty array
    """
    
    N = len(prop_steps)
    d = len(prop_steps[0])
    
    prop_intervals = empty((N,N,d,d), dtype = complex)
    
    for i in range(N):
        # initialize the diagonal
        prop_intervals[i,i] = prop_steps[i]
        
        for j in range(i+1,N):
            prop_intervals[i,j] = prop_intervals[i,j-1]@prop_steps[j]
    
    
    return prop_intervals
            
    

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