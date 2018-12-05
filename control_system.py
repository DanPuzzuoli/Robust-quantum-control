#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 22:00:31 2018

@author: Daniel Puzzuoli


Description:
    This file defines the "control_system" object, which stores the drift and
    control generators, and has some simple functionality for computing the full
    generator from control amplitudes.

Notes:
    At the moment I'm not sure exactly what will go in here as functionality grows,
    but it seems natural to also include functions for creating derived control
    systems. E.g. given a control system, create the corresponding system for 
    computing decoupling a particular matrix.
    
To do:
    - One possible direction for this object is to design it to handle
    redundancies in the computation. I.e. In doing searching for a decoupling
    pulse, and computing derivatives, there are technically many linear/control 
    systems involved, but often they will contain redundant information. E.g.
    in the most extreme case the "derivative" control system completely contains
    the base level control system. So, one possibility is to have a network
    of control system objects that will draw on eachother for computations to
    try to automate the problem of eliminating redundancies
    
"""

from numpy import tensordot,kron,identity, array
from utb_matrices import decoupling_gen

class control_system:
    
    
    def __init__(self, drift, control_generators):
        """
        The constructor for the class.
        
        Parameters
        ----------
        drift : numpy.array
            The drift generator, assumed shape is (dim,dim)
        control_generators : numpy.array
            The control generators, assumed shape is either (c, dim, dim),
            or (dim, dim) in the case of a single generator, which will then
            be converted to the shape (1, dim, dim)
    
        """
        
        # set the drift
        self.drift = drift
        
        # Some basic type handling for specifications of control parameters
        # allows specification as a 3d array, a list of 2d arrays, or as 
        # a single 2d array.
        # ultimately want it to be specified as a 3d array
        if type(control_generators) is list:
            self.control_generators = array(control_generators)
        else:    
            if len(control_generators.shape) == 2:
                self.control_generators = array([control_generators])
            else:
                self.control_generators = control_generators
            
        self.system_dim = len(drift)
        self.control_dim = len(control_generators)
        
    def generator(self,amps):
        """
        Computes the full generator for a single time step
        
        Parameters
        ----------
        amps : numpy.array
            a 1 dimensional array whose length is assumed to be equal to
            control_dim
    
        Returns
        -------
        numpy.array - shape (system_dim, system_dim)
        
        drift + sum_i amps[i]*control_generators[i]
        """
        return self.drift + tensordot(amps, self.control_generators, axes = (0,0))
    
    def generator_list(self, amp_list):
        """
        Computes the full generator for multiple time steps
        
        Parameters
        ----------
        amp_list : numpy.array
            a 2 dimensional array with the length of the second dimension
            assumed to be equal to self.control_dim
    
        Returns
        -------
        numpy.array - shape (len(amp_list), system_dim, system_dim)
        
        output[j] = drift + sum_i amps[j,i]*control_generators[i]
        """
        return self.drift + tensordot(amp_list, self.control_generators, axes = (1,0))
    
    
    def decoupling_system(self, Alist):
        """
        Given a list of matrices to decouple, returns a new control system
        for the decoupling problem specified by Alist
        
        Parameters
        ----------
        Alist : a list of 2d numpy arrays, or a single 2d numpy array
            The numpy arrays are assumed to have shape (system_dim, system_dim).

        Returns
        -------
        A new control_system object corresponding to the system that computes
        the decoupling terms
        
        """
        
        if type(Alist) is not list:
            k = 2
        else:
            k = len(Alist) + 1
        
        new_drift = decoupling_gen(self.drift, Alist)
        new_c_generators = kron(identity(k),self.control_generators)
        
        return control_system(new_drift,new_c_generators)