#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:06:26 2018

@author: dpuzzuol

This file just contains the function "matmult", whose input/output format
is exactly the same as numpy.matmul. The reason for this function is to easily
chang ebetween methods for multiplying ndarrays of matrices depending on what
is fastest for the current version of numpy. At the time of writing, simply
looping through and using dot( , ) on every pair of matrices seems to be
faster
"""

from numpy import dot,empty_like

def matmult(a,b):
    # this is kind of silly, couldn't find out how to index arrays using a list
    # or tuple in the way I wanted.
    # only works up to ndim 6
    out = empty_like(a)
    
    if out.ndim == 2:
        out = dot(a,b)
    elif out.ndim == 3:
        for i in range(a.shape[0]):
            out[i] = dot(a[i], b[i])
    elif out.ndim == 4:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i,j] = dot(a[i,j],b[i,j])
    elif out.ndim == 5:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    out[i,j,k] = dot(a[i,j,k],b[i,j,k])
    elif out.ndim == 6:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    for m in range(a.shape[3]):
                        out[i,j,k,m] = dot(a[i,j,k,m],b[i,j,k,m])
    
    return out
