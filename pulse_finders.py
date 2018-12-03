#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:21:48 2018

@author: dpuzzuol

Description: 
    This file containts functions that set up optimizations given objective
    functions for the rest of the code. 
    
    For example: 
        -   Modifies objectives from the other files, which act/work with 2d
            arrays of amplitudes, to act/work with 1d arrays, which the scipy
            optimizers require
        -   Modifies objectives to output updates on the current value every
            so many calls
        -   Calls the scipy minimizers

Notes:
    -   Currently ctrl_shape is being passed, but I suppose it could be deduced
        from initial_guess
    -   In the future may have multiple optimization algorithms
    -   At a future date, will also include further functionality for managing
        optimizations; e.g. automatically quiting searches that have run
        for too long and restarting, running multiple searches in parallel and
        choosing best one...
"""

from scipy.optimize import minimize
from time import time


def find_pulse_bfgs(obj, ctrl_shape, initial_guess, update_rate = 100):
    """
    Given an objective function, a tuple representing the shape of the control,
    an initial guess, and a desired rate of printed updates, runs the BFGS
    algorithm to minimize obj.
    
    Parameters
    ----------
    obj : function
        A function representing the objective function f, assumed to take in 2d 
        numpy arrays with shape given by ctrl_shape, and output a tuple with 
        f(x),jac(f)(x), i.e. the function evaluated at the input, and the
        jacobian of f at the input, where the jacobian is a 2d array also with
        shape ctrl_shape
    ctrl_shape : tuple
        The shape of the input of obj
    initial_guess : numpy.array
        A 2d array with shape ctrl_shape
    update_rate : int or None
        The desired rate at which the current value of obj is printed. (i.e.
        if update_rate == 100, then the value of obj is updated every 100 calls)
        If a value of None is given, no updates are reported

    Returns
    -------
    The output of the BFGS optimization of obj with the starting point
    initial_guess
    """
    
    # set the start time
    start = time()
    
    # we modify the objective function to take in vectorized inputs (which
    # scipy optimizers require), and return vectorized jacobians
    mod_obj = vectorized_function(obj, ctrl_shape, deriv=1)
    
    # modify the objective to print updates, if desired 
    if update_rate is not None:
        mod_obj = updating_function(mod_obj, update_rate)
    
    
    print('Optimizing pulse...')
    # run the optimization
    result = minimize(mod_obj, initial_guess.flatten(), method='BFGS', jac=True, options={'disp': True})
    # reshape the point the optimizer ends on to be the correct shape
    result.x = result.x.reshape(ctrl_shape)
    
    # record the end time and report the total time taken
    end = time()
    print('Total time taken: ' + str(end-start))
    
    return result

def vectorized_function(f, ctrl_shape, deriv = 0):
    """
    Given a function that takes in numpy arrays of shape ctrl_shape, and returns
    either f(x) or a tuple f(x), jac(f)(x), returns a new function that takes
    in vectorized input, and if f returns a jacobian, also modifies it to 
    return the flattened jacobian
    
    Parameters
    ----------
    f : function
        The function,whose inputs are 2d arrays with shape ctrl_shape. Assumed
        to have different return value forms depending on the input deriv
    ctrl_shape : tuple
        The shape of the input of f
    deriv : int
        Representing the number of derivatives f returns. If deriv == 0, f 
        just returns f(x). If deriv == 1, f returns a tuple f(x), jac(f)(x),
        where jac(f)(x) is a numpy array of shape ctrl_shape

    Returns
    -------
    A modified version of f as described above
    """
    
    if deriv == 0:
        # if its just a scalar function, only need to reshape the input
        return lambda x: f(x.reshape(ctrl_shape))
    
    elif deriv == 1:
        # if its a function that returns a tuple (f(x), jac(f)(x)), 
        # reshape input, and flatten returned jacobian
        def vecf(x):
            val, jac = f(x.reshape(ctrl_shape))
            return val, jac.flatten()
        
        return vecf

def updating_function(f, update_rate):
    """
    Given a function f, and an int update_rate, returns a function upd_f that
    behaves the same as f, but prints a message reporting the value of f every
    update_rate calls
    
    Parameters
    ----------
    f : function
        The function to be updated. It is assumed to either return a scalar,
        or to return a tuple. In the case of a tuple, the "value" of f to be
        reported is assumed to be the first entry, which is in line with
        how we assume objectives to behave: they return either a scalar,
        or a tuple of derivatives of increasing order, with the first entry
        the 0^{th} order derivative (i.e. the function value)
    update_rate : int
        The rate at which updates should be printed

    Returns
    -------
    A new function f as described above
    """
    
    # set up the call counter
    calls = 0
    
    # define the new function
    def upd_f(x):
        # give the function access to the call counter, and increment it
        # every call
        nonlocal calls
        calls = calls + 1
        
        #compute f
        output = f(x)
        
        # if its time to give an update, report the update depending on the
        # format of the output of f
        if calls % update_rate == 0:
            if type(output) == tuple:
                print('Value at evaluation ' + str(calls) + ': '+  str(output[0]))
            else:
                print('Value at evaluation ' + str(calls) + ': '+  str(output))
        
        return output
    
    return upd_f