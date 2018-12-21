#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:43:34 2018

@author: Daniel Puzzuoli
"""
from numpy import zeros,matmul
from numpy.random import rand
from time import time
from timeit import timeit


d = 2
N = 1500
k = 13 # power to compute

# create random 3x3 block generators. The first index is a list of generators,
# and each generator is represented as 3d array of the form (G,A,B), where G,A,
# and B are d x d arrays.
generators = rand(N,3,d,d)

# powers are the same as above except they are 3d arrays of the form 
# (X,C,D,E)
# powers is initialized to contain (G,A,B,0), i.e. the first power
powers = zeros( (N,5,d,d) )
powers[:, 0:3,:,:] = generators.copy()

print('Computing using structure...')
start = time()
for m in range(1,k):
    powers[:,4] = matmul(generators[:,0],powers[:,4]) + matmul(generators[:,2],powers[:,1])
    powers[:,3] = matmul(generators[:,0],powers[:,3]) + matmul(generators[:,1],powers[:,2])
    powers[:,2] = matmul(generators[:,0],powers[:,2]) + matmul(generators[:,2],powers[:,0])
    powers[:,1] = matmul(generators[:,0],powers[:,1]) + matmul(generators[:,1],powers[:,0])
    powers[:,0] = matmul(generators[:,0],powers[:,0])

end = time()
print('Done, total time: ' + str(end - start))

# next, computing using full matrices
full_generators = zeros((N,3*d,3*d))
# set the diagonals
full_generators[:,0:d,0:d] = generators[:,0,:,:].copy()
full_generators[:,d:2*d,d:2*d] = generators[:,0,:,:].copy()
full_generators[:,2*d:3*d,2*d:3*d] = generators[:,0,:,:].copy()
#set the off diagonals
full_generators[:,0:d,d:2*d] = generators[:,1,:,:].copy()
full_generators[:,d:2*d,2*d:3*d] = generators[:,2,:,:].copy()

# construct the same generators but with the off diagonals swapped
full_generators2 = full_generators.copy()
full_generators2[:,0:d,d:2*d]= full_generators[:,d:2*d,2*d:3*d].copy()
full_generators2[:,d:2*d,2*d:3*d]= full_generators[:,0:d,d:2*d].copy()


powers2 = full_generators.copy()
powers3 = full_generators2.copy()

print('Computing using direct multiplication...')
start = time()
for m in range(1,k):
    powers2 = matmul(full_generators,powers2)
    powers3 = matmul(full_generators2,powers3)
end = time()
print('Done, total time: ' + str(end - start))

print('accuracy in diag1: ' + str( (abs(powers[:,0] - powers2[:,0:d,0:d])).max() ))
print('accuracy in diag2: ' + str( (abs(powers[:,0] - powers2[:,d:(2*d),d:(2*d)])).max() ))
print('accuracy in diag1: ' + str( (abs(powers[:,0] - powers2[:,(2*d):(3*d),(2*d):(3*d)])).max() ))
print('accuracy in offdiag1: ' + str( (abs(powers[:,1] - powers2[:,0:d,d:(2*d)])).max() ))
print('accuracy in offdiag2: ' + str( (abs(powers[:,2] - powers2[:,d:2*d,2*d:3*d])).max() ))
print('accuracy in offdiag3: ' + str( (abs(powers[:,3] - powers2[:,0:d,2*d:(3*d)])).max() ))
print('accuracy in offdiag3-prime: ' + str( (abs(powers[:,4] - powers3[:,0:d,2*d:(3*d)])).max() ))