#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:44:34 2018

@author: Daniel Puzzuoli

Descriptions:
    A place to define some matrices
"""

from numpy import array

def pauliX():
    return array([[0,1.],[1.,0]])

def pauliY():
    return array([[0,-1j],[1j,0]])

def pauliZ():
    return array([[1.,0],[0,-1.]])