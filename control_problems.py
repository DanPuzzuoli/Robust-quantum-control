#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:48:34 2018

@author: Daniel Puzzuoli

Description:
    A place to set up and store control problems
"""

from numpy import real, zeros, ones, identity
import control_systems as css
from numpy.random import rand
import constraints as cons
from evolve_system import evolve_system
from scipy.optimize import minimize
from objective_functions import grape_objective, zero_block_objective
import hamiltonians as h

def universal_id_xy(initial_guess = None):
    '''
    This is setting up a control problem:
        - system: 0 drift, and x,y control
        - universal decoupling
        -implement an identity gate
    '''
    
    '''
    #this spits out a skewed XY4
    N = 4
    dt = 0.5
    power_ub = [1,1]
    power_lb = [-1,-1]
    power_tol = 0.0005
    
    change_b = [2,2]
    change_tol = 0.005
    '''
    
    xy_sys = css.XandY() # get the control system forX and Y control
    
    # get the derived control system for computing both the unitary, as well
    # as the first order decoupling terms for X, Y, and Z
    dec_sys = xy_sys.decoupling_system([h.pauliX(), h.pauliY(), h.pauliZ()])
    
    
    # a smoothed version of XY4
    # N = 80 worked, N=75 doesn't seem to
    N=80 # number time steps
    dt = 0.05 # time step length
    power_ub = [1,1] # upper bounds on control amplitudes
    power_lb = [-1,-1] # lower bounds on control amplitudes
    power_tol = 0.0005 # tolerance for when penalty function for control amplitude power kick in
    
    change_b = [0.05,0.05] # each contorl amplitude is restricted to change by less than 0.05
    change_tol = 0.0005 # tolerance for enforcing rate of change
    
    
    # set the target gate to the identity
    Utarget = identity(2)
    
    # set a variable for the shape of the array of control amplitudes
    ctrl_shape = (N, len(xy_sys.control_generators))
    
    
    #set up the objective function. The optimizer assumes x is a 1d array,
    # so we always need to reshape it as our code deals with a 2d array
    def obj(x):
        x = x.reshape(ctrl_shape)
        
        # evolve the system that computes both the final unitary as well
        # as decoupling terms
        prop = evolve_system(dec_sys, x, dt, deriv = 1)
        
        #extract the final unitary, as well as the array of derivatives
        # with respect to control amplitudes
        Ufinal = prop[0][0:2,0:2]
        Uderiv = prop[1][:,0:2,0:2]
        
        # compute the grape objective; the fidelity to the target gate
        # as well its jacobian
        g,gp = grape_objective(Utarget, (Ufinal, Uderiv), deriv = 1)
        
        # compute the norms of the decoupling terms, as well as their
        # derivatives
        decx,decxp= zero_block_objective(prop, 2, 0,1, deriv = 1)
        decy,decyp= zero_block_objective(prop, 2, 1,2, deriv = 1)
        decz,deczp= zero_block_objective(prop, 2, 2,3, deriv = 1)
        
        # add together the decoupling objectives and derivatives
        dec = decx+ decy+decz
        decp = decxp + decyp + deczp

        # finally, compute penalties on the pulse shape        
        power = cons.mono_power_constraint(x,power_lb,power_ub,2,power_tol,0)
        powerp = cons.mono_power_constraint(x,power_lb,power_ub,2,power_tol,1)
        change = cons.mono_smoothness_constraint(x,change_b,2,change_tol,0)
        changep = cons.mono_smoothness_constraint(x,change_b,2,change_tol,1)
        
        # return a sum of all of the above computed parts to the objectives,
        # as well as their derivatives
        return real(-g+dec +power+change/20),real((-gp+decp+powerp+changep/20).flatten())

    # create a random initial guess
    if initial_guess == None:
        x0=rand(*ctrl_shape)*change_b 
    else:
        x0 = initial_guess.copy()[0:N]
    #x0 = ones(ctrl_shape)
    
    
    # run the minimizer and return the results
    res = minimize(obj, x0.flatten(), method='BFGS', jac=True, options={'disp': True})
    res.x = res.x.reshape(ctrl_shape)
    return res
    
def x_sys_dec_z():
    '''
    This is a control problem for: a system with only x control and no drift,
    create a Pauli X gate while also decoupling Z to first order.
    '''
    
    # get the control system with only X control
    x_sys = css.onlyX()
    
    # generate the control system that also compute Z decoupling term
    dec_z_sys = x_sys.decoupling_system([h.pauliZ()])
    
    
    #number of time steps and time step length
    N = 157
    dt = 0.0125
    
    # target unitary is pauli X
    Utarget = h.pauliX()
    
    # set bounds on the control amplitude power, and tolerance
    power_ub = 1
    power_lb = -1
    power_tol = 0.05
    
    # bounds on the control amplitude rate of change and tolerance
    change_b = 0.025
    change_tol = 0.005
    
    ctrl_shape = (N, len(dec_z_sys.control_generators))
    
    # create the objective for the search
    def obj(x):
        x = x.reshape(ctrl_shape)
        
        # fora given control value, propagate the decoupling system, and compute
        # derivatives
        prop = evolve_system(dec_z_sys, x, dt, deriv = 1)
        
        # extract the final unitary and its derivatives
        Ufinal = prop[0][0:2,0:2]
        Uderiv = prop[1][:,0:2,0:2]

        # compute the target unitary objective and its derivatives
        g,gp = grape_objective(Utarget, (Ufinal,Uderiv), deriv = 1)
        
        # compute the decoupling objective and its derivatives
        dec,decp= zero_block_objective(prop, 2, 0,1, deriv = 1)

        #compute the penalties for pulse shape        
        power = cons.mono_power_constraint(x,power_lb,power_ub,2,power_tol,0)
        powerp = cons.mono_power_constraint(x,power_lb,power_ub,2,power_tol,1)
        change = cons.mono_smoothness_constraint(x,change_b,2,change_tol,0)
        changep = cons.mono_smoothness_constraint(x,change_b,2,change_tol,1)
        
        # return a sum of the various objectives 
        return real(-g+dec+power+change/20),real((-gp+decp+powerp+changep/20).flatten())

    # select a random starting point
    x0=rand(*ctrl_shape)#*change_b 
    
    #run the optimization and return the results
    res = minimize(obj, x0.flatten(), method='BFGS', jac=True, options={'disp': True})
    res.x = res.x.reshape(ctrl_shape)
    return res
