#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:26:40 2018

@author: dpuzzuol
"""

from numpy import empty
from numpy.random import rand
from utb_matrices import exp_deriv
from comm_derivs import exp_deriv_comm
from time import time

N=5000
d = 8
k=30

A = rand(N,d,d)
B = rand(N,d,d)

U = empty((N,d,d), dtype = complex)
Ud= empty((N,d,d), dtype = complex)
print('computing via exponentials')
start = time()
for n in range(N):
    U[n],Ud[n] = exp_deriv(A[n],B[n])

end = time()
print('Total time: ' + str(end-start) + ', average time: ' + str((end-start)/N) )


V = empty((N,d,d), dtype = complex)
Vd= empty((N,d,d), dtype = complex)
print('computing via series of order ' + str(k))
start = time()
for n in range(N):
    V[n],Vd[n] = exp_deriv_comm(A[n],B[n],k)

end = time()
print('Total time: ' + str(end-start) + ', average time: ' + str((end-start)/N) )

print('Max difference: ' + str((Ud - Vd).max()))