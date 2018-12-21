#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:13:48 2018

@author: dpuzzuol
"""

from scipy.linalg import expm 

def exp_deriv_comm(A,B,k):
    U = expm(A)
    
    D = B
    
    if k ==0:
        return U, D@U
    
    C = B
    for n in range(1,k+1):
        C = commutator(A,C)/(n+1)
        D = D + C
    
    return U, D@U
    
def commutator(A,B):
    return A@B - B@A