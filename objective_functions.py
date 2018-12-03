#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:25:06 2018

@author: Daniel Puzzuoli

Description:
    Currently a working file, but this is a place to put functions for
    creating objectives

Notes:
    
"""

from numpy import real,empty,ndarray
from utb_matrices import get_block

def zero_block_objective(prop,block_dim, row, col,deriv=0):
    """
    the linear function f is just the extraction of a block
    """
    def f(x):
        return get_block(x, block_dim,row,col)
    
    return generalized_grape_objective(f, prop, deriv=deriv)

def grape_objective(Utarget, prop, deriv=0):
    """
    the standard grape objective, where the linear function is
    f(x)= hs_inner(Utarget, x)
    """
    def f(x):
        return hs_ip(Utarget, x)
    
    return generalized_grape_objective(f, prop, deriv=deriv)

def generalized_grape_objective(f, prop, deriv = 0):
    """
    Given a linear function, f, computes the Frobenius norm squared of
    f(prop). (Note that technically the function doesn't need to be linear
    for this function to work, but the form of the derivatives is assumes
    f is linear [and so commutes with differentiation]) We assume that
    f maps square arrays to square arrays. Note that even if f is a scalar,
    it should be output as an array
    
    If deriv == 0, prop is assumed to be an array that the objective is computed
    on
    
    If deriv == 1, prop is assumed to be a tuple, with the first entry the
    prop, and the second entry a 4d array, where...
    """
    
    if deriv == 0:
        x = f(prop)
        
        return real(hs_ip(x,x))
    elif deriv == 1:
        U,Up = prop
        
        #compute the value of f
        x = f(U)
        
        # get the number of time steps and control parameters
        N = len(Up)
        ctrl_dim = len(Up[0])
        
        
        # compute the jacobian
        jacobian = empty((N,ctrl_dim), dtype = complex)
        for tstep in range(N):
            for ctrl_i in range(ctrl_dim):
                jacobian[tstep,ctrl_i] = 2*real(hs_ip(x,f(Up[tstep,ctrl_i])))
        
        return real(hs_ip(x,x)), jacobian
    
        

    
def hs_ip(a,b):
    """
    Compute the Hilbert-Schmidt inner product of a and b (two matrices given
    as numpy arrays), using the convention that it is conjugate linear in
    the first argument
    """
    
    # Check if the inputs are not arrays. If they are not we assume they
    # are scalars
    if (type(a) != ndarray) & (type(b) != ndarray):
        return a.conjugate() * b
        
    # if they are ararys, return the usual Hilbert-Schmidt inner product
    return (a.conjugate().transpose()@b).trace()
