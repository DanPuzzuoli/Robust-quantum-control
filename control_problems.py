#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:48:34 2018

@author: Daniel Puzzuoli

Description:
    A place to set up and store control problems
"""

from numpy import real, identity
import prebuilt_systems as css
from numpy.random import rand
import constraint_functions as cons
from evolve_system import evolve_system
from objective_functions import grape_objective, zero_block_objective
import hamiltonians as h
from pulse_finders import find_pulse_bfgs

def universal_id_xy(initial_guess = None):
    '''
    This is setting up a control problem:
        - system: 0 drift, and x,y control
        - universal decoupling
        -implement an identity gate
    '''
    
    xy_sys = css.XandY() # get the control system forX and Y control
    
    # get the derived control system for computing both the unitary, as well
    # as the first order decoupling terms for X, Y, and Z
    dec_sys = xy_sys.decoupling_system([h.pauliX(), h.pauliY(), h.pauliZ()])
    
    
    # a smoothed version of XY4
    # N=76 -4+10^[-10], doesn't find it every time but seems to work pretty
    # often
    # N=75 -3.99655
    N=76 # number time steps
    dt = 0.05 # time step length
    power_ub = [1,1] # upper bounds on control amplitudes
    power_lb = [-1,-1] # lower bounds on control amplitudes
    power_tol = 0.0005 # tolerance for when penalty function for control amplitude power kick in
    
    change_b = [0.05,0.05] # each control amplitude is restricted to change by less than 0.05
    change_tol = 0.0005 # tolerance for enforcing rate of change
    
    
    # set the target gate to the identity
    Utarget = identity(2)
    
    # set a variable for the shape of the array of control amplitudes
    ctrl_shape = (N, xy_sys.control_dim)
    
    #num = 0
    def obj(x):
        #nonlocal num
        #num = num+1
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
        
        shape,shaped = cons.mono_objective(x, power_lb, power_ub, power_tol, change_b, change_tol, deriv = 1)
        
        
        #if num % 100 == 0:
        #    print('Value at ' + str(num) + ' iterations: '+  str(real(-g+dec+shape)))
        
        #return real(-g+dec+power+change/20),real(-gp+decp+powerp+changep/20)
        return real(-g + dec + shape/100), real(-gp + decp + shaped/100)
    
    
    
    res = find_pulse_bfgs(obj,ctrl_shape, rand(*ctrl_shape)*change_b)
    return res

    
    


def x_sys_dec_z():
    '''
    This is a control problem for: a system with only x control and no drift,
    create a Pauli X gate while also decoupling Z to first order.
    '''
    
    # get the control system with only X control
    x_sys = css.onlyX()
    
    # generate the control system that also compute Z decoupling term
    dec_z_sys = x_sys.decoupling_system(h.pauliZ())

    
    
    # So far, shortest time with many 9s is 152, though doesn't find it
    # every time. Using N=151 has so far resulted in the search terminating
    # every time at a value of ~ -3.999824
    N = 152
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
    
    def obj(x):
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
        
        shape,shaped = cons.mono_objective(x, power_lb, power_ub, power_tol, change_b, change_tol, deriv = 1)
        
        return real(-g + dec+shape/20),real(-gp + decp+ shaped/20)
    
    
    ctrl_shape = (N, len(dec_z_sys.control_generators))
    
    res = find_pulse_bfgs(obj, ctrl_shape,rand(*ctrl_shape))
    return res