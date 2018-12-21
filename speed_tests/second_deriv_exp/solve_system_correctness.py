#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:53:45 2018

@author: dpuzzuol

testing that the solving works
"""

from numpy.linalg import solve,norm
from numpy.random import rand
from utb_derivative_system import utb_derivative_system

der = 2
trials = 50

N = 100
ds = 2
dc = 2

max_diff = 0.0

for k in range(trials):

    G1 = rand(N,ds,ds)
    G2 = rand(N,ds,ds)
    A1 = rand(dc,ds,ds)
    A2 = rand(dc,ds,ds)
    if der == 1:
        B1 = None
        B2 = None
    elif der == 2:
        B1 = rand(N,dc,dc,ds,ds)
        B2 = rand(N,dc,dc,ds,ds)
    
    obj1 = utb_derivative_system(G1, A1,B1, deriv = der)
    obj2 = utb_derivative_system(G2, A2,B2, deriv = der)
    
    
    obj1 = utb_derivative_system(G1, A1, deriv = 1)
    obj2 = utb_derivative_system(G2, A2, deriv = 1)
    mat1 = obj1.matrixformfull()
    mat2 = obj2.matrixformfull()
    
    objsol = obj1.solve(obj2)
    matsol = solve(mat1, mat2)
    
    for i in range(N):
        for j in range(dc):
            for k in range(dc):
                rel_diff = norm(objsol.matrixform(i,j,k)- matsol[i,j,k])/norm(matsol[i,j,k])
                if rel_diff > max_diff:
                    max_diff = rel_diff


print(str(max_diff))
