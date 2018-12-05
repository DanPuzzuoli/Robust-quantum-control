#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:32:29 2018

@author: dpuzzuol

Description: 
    This file contains functions for defining some standard control
    systems

Note:
    Should change name of this file.
    
To do:
    Change name
"""
from math import pi
import hamiltonians as h
from numpy import zeros,array
from control_system import control_system

def onlyX():
    """
    Defines a control system that is 2 dimensional, has no drift, and
    has only Pauli X control
    """
    drift = zeros( (2,2) )
    c_gen = -1j*pi*h.pauliX()
    
    return control_system(drift,c_gen)

def XandY():
    """
    Defines a control system that is 2 dimensional, has no drift, and
    has Pauli X and Y control
    """
    drift = zeros( (2,2) )
    c_gen = array([-1j*pi*h.pauliX(), -1j*pi*h.pauliY()])
    
    return control_system(drift,c_gen)