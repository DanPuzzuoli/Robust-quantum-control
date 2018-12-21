#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:58:27 2018

@author: dpuzzuol
"""

from numpy import matmul
from matmult import matmult
from numpy.random import rand
from utb_derivative_system import utb_derivative_system
from timeit import timeit

trials = 50
includeB = True

der = 2

N = 100
ds = 4
dc = 2

objtime = 0.0
mattime = 0.0
matmultime=0.0

for k in range(trials):

    G1 = rand(N,ds,ds)
    G2 = rand(N,ds,ds)
    A1 = rand(dc,ds,ds)
    A2 = rand(dc,ds,ds)
    
    if (der == 1) or (includeB is False):
        B1 = None
        B2 = None
    elif (der == 2) and (includeB is True):
        B1 = rand(N,dc,dc,ds,ds)
        B2 = rand(N,dc,dc,ds,ds)
    
    obj1 = utb_derivative_system(G1, A1,B1, deriv = der)
    obj2 = utb_derivative_system(G2, A2,B2, deriv = der)
    mat1 = obj1.matrixformfull()
    mat2 = obj2.matrixformfull()
    
    objtime = objtime + timeit("obj1.matmul(obj2)", number=1, globals=globals())
    matmultime=matmultime + timeit("matprod = matmul(mat1,mat2)", number=1, globals=globals())
    mattime = mattime + timeit("matprod = matmult(mat1,mat2)", number=1, globals=globals())
    
    
    
print("obj time: " + str(objtime/trials))
print("mat time: " + str(mattime/trials))
print("matmul time: " + str(matmultime/trials))
print("ratio: " + str(mattime/objtime))
