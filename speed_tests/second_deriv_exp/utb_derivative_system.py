#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 08:56:26 2018

@author: dpuzzuol

Notes:
    This object has various behaviours depending on the value of deriv, and
    could also benefit from type checking. May want to add this into the
    constructor, as well as in the functions for which these objects interact
    
    In computing the B[i,j,k] in matmul, consider how else matmul could be
    called to do this without having to use loops. Also, are there any name
    collision issues with having a method also called matmul? Doesn't seem to
    be at the moment.
"""

from matmult import matmult
from numpy import empty,empty_like,zeros,block, identity, amax, absolute
from numpy.linalg import solve

class utb_derivative_system(object):
    def __init__(self, G, A, B = None, deriv=2):
        """
        Initialize. B takes default value None as it will be common to
        initialize to a set of matrices for which the top right is 0
        
        Currently not verifying correct shape
        
        
        -G is a 3d numpy array of shape (N,ds,ds)
        -A is either a 3d array of shape (dc,ds,ds), or a 4d array of shape
           (N,dc,ds,ds). If specified as a 3d array, it is transformed to the
           form of the 4d array
        -B is assumed to be either None, or an array of shape (N, dc, dc, ds, ds)
         How it is handled depends on what deriv is. If deriv==2, the None input 
         is equivalent to setting B=zeros((N,dc,dc,ds,ds)).
         if deriv == 1, then B is set to None regardless of the passed value,
         and plays no role in future computations.
        """
        # set G, and extract dimensions
        self.G = G
        self.N,self.ds = G.shape[0:2]
        
        # different behavior depending on the number of indices of A
        if A.ndim == 4:
            # if 4, assume that it is the correct shape, set A,
            # extract control dimension
            self.A = A
            self.dc = A.shape[1]
        elif A.ndim == 3:
            # if 3, we assume shape (dc, ds, ds), and we need to copy A
            # N times into an (N,dc,ds,ds)-array
            self.dc = A.shape[0]
            Anew = empty((self.N,self.dc, self.ds,self.ds), dtype = A.dtype)
            for i in range(self.N):
                Anew[i] = A
            self.A = Anew
        
        self.deriv= deriv
        
        # if deriv==1, set B = None as whatever is passsed will play
        # no role in future computations (could put a check here)
        if deriv == 1:
            self.B = None
        elif deriv==2:
            self.B = B
        
        # Note: not sure what it makes sense to set the ``shape'' to. Need to
        # think about this. Could have a shape and a matshape attribute
        self.matshape = (self.N,self.dc,self.dc,(self.deriv+1)*self.ds,(self.deriv+1)*self.ds)

    
    def matmul(self, other):
        """
        For multiplying two of these systems.
        
        Assumptions:
            the two systems have the same shape and their derivs are equal
        """
        
        # First, do the computations that are common to the deriv==1 and 
        # deriv=2 cases
        
        # compute new G matrices
        Gnew = matmult(self.G, other.G)
        
        # new A matrices
        Anew = empty_like(self.A)
        for j in range(self.A.shape[1]):
            Anew[:,j] = matmult(self.G, other.A[:,j]) + matmult(self.A[:,j], other.G)
        
        #if deriv ==1, we are done
        if self.deriv == 1:
            return utb_derivative_system(Gnew, Anew, None, 1)
        
        # if deriv==2, compute the B matrices. How to do this depends on if
        # self.B is None or other.B is None (or both)
        
        # create the B matrices         
        Bnew = empty((self.N, self.dc, self.dc, self.ds, self.ds), dtype = self.A.dtype)
        
        # Each B matrix is the sum of potentially 3 multiplications.
        # loop through j,k indices
        for j in range(self.A.shape[1]):
            for k in range(self.A.shape[1]):
                # first, compute the term that doesn't depend on the status
                # of self.B or other.B
                Bnew[:,j,k] = matmult(self.A[:,j], other.A[:,k])
                
                # next, compute the terms depending on if self.B or other.B
                # are not None
                if self.B is not None:
                    Bnew[:,j,k] = Bnew[:,j,k] + matmult(self.B[:,j,k], other.G)
                
                if other.B is not None:
                    Bnew[:,j,k] = Bnew[:,j,k] + matmult(self.G,other.B[:,j,k])
        
        return utb_derivative_system(Gnew, Anew, Bnew, 2)

    
        
    def scalarmul(self, a):
        """
        multiply all matrices by a scalar. Behaviour depends on if B is None
        """
        if self.B is None:
            return utb_derivative_system(a*self.G,a*self.A,None,self.deriv)
        else:
            return utb_derivative_system(a*self.G,a*self.A,a*self.B,self.deriv)
    
    def scalaradd(self,a):
        """
        adding scalars
        only G needs to change, as it is on the diagonal
        """
        return utb_derivative_system(a*identity(self.ds) + self.G,self.A,self.B,self.deriv)
        
    
    def linearcombo(self, other, a,b):
        if self.deriv ==1:
            return utb_derivative_system(a*self.G + b*other.G, a*self.A + b*other.A, None, 1)
        
        # if self.deriv == 2:
        
        if (self.B is None) and (other.B is None):
            return utb_derivative_system(a*self.G + b*other.G, a*self.A + b*other.A, None, 2)
        elif (self.B is None) and (other.B is not None):
            return utb_derivative_system(a*self.G + b*other.G, a*self.A + b*other.A, b*other.B, 2)
        elif (self.B is not None) and (other.B is None):
            return utb_derivative_system(a*self.G + b*other.G, a*self.A + b*other.A, a*self.B, 2)
        else:
            return utb_derivative_system(a*self.G + b*other.G, a*self.A + b*other.A, a*self.B + b*other.B, 2)
    
    def matadd(self,other):
        """
        add a system
        """
        return self.linearcombo(other, 1, 1)
    
    def matsub(self,other):
        """
        subtract a system
        """
        return self.linearcombo(other,1, -1)

    
    
    def matrixform(self,i,j,k = None):
        """
        As this object is technically storing matrices indexed by i,j,k,
        this function returns the full form of the matrix (not sure if this
        is necessary,but is nice for testing)
        
        If self.deriv==1, k should not be specified, or if it is it will be
        ignored
        
        Again, should check if k is specified if deriv==2
        """        
        
        z = zeros((self.ds,self.ds))
        
        if self.deriv==1:
            return block([ [self.G[i], self.A[i,j]], 
                           [z, self.G[i]]])
        
        # if self.deriv ==2, handle the B cases
        
        if self.B is None:
            return block([[self.G[i], self.A[i,j], z], 
                          [z, self.G[i], self.A[i,k]], 
                          [z, z, self.G[i]]])
        else:
            return block([[self.G[i], self.A[i,j], self.B[i,j,k]], 
                          [z, self.G[i], self.A[i,k]], 
                          [z, z, self.G[i]]])
    
    def matrixformfull(self):
        
        full = empty(self.matshape, dtype = self.A.dtype)
        
        if self.deriv==1:
            for i in range(self.N):
                for j in range(self.dc):
                    full[i,j] = self.matrixform(i,j)
        elif self.deriv==2:
            for i in range(self.N):
                for j in range(self.dc):
                    for k in range(self.dc):
                        full[i,j,k] = self.matrixform(i,j,k)
        
        return full
        
    
    def maxcolnorm(self):
        if self.B is None:
            maxnorm = 0
            for i in range(self.N):
                for j in range(self.dc):
                    n = amax(absolute(block([[self.A[i,j]], [self.G[i]]])).sum(axis=-2))
                    if n > maxnorm:
                        maxnorm = n
            
            return maxnorm
        else:
            maxnorm = 0
            for i in range(self.N):
                for j in range(self.dc):
                    for k in range(self.dc):
                        n = amax(absolute(block([[self.B[i,j,k]],[self.A[i,j]], [self.G[i]]])).sum(axis=-2))
                        
                        if n > maxnorm:
                            maxnorm = n
            return maxnorm
        
        
    
    def solve(self,other):
        """
        solving the matrix system (self)X = other
        """
        
        #first find diagonals of the solution
        X = solve(self.G,other.G)
        
        #next the first off diagonals
        Y = empty_like(self.A)
        for j in range(self.dc):
            Y[:,j] = solve(self.G,other.A[:,j] - matmult(self.A[:,j], X) )
        
        # if deriv==1, that's all we need
        if self.deriv==1:
            return utb_derivative_system(X,Y,None,1)
        
        #next, solve for the top right
        Z = empty((self.N,self.dc,self.dc,self.ds,self.ds), dtype = self.A.dtype)

        for j in range(self.dc):
            for k in range(self.dc):
                # different behaviors on if the Bs are None
                if (self.B is None) and (other.B is None):
                    Z[:,j,k] = solve(self.G, -matmult(self.A[:,j],Y[:,k]) )
                elif (self.B is None) and (other.B is not None):
                    Z[:,j,k] = solve(self.G, other.B[:,j,k]-matmult(self.A[:,j],Y[:,k]) )
                elif (self.B is not None) and (other.B is None):
                    Z[:,j,k] = solve(self.G, -matmult(self.B[:,j,k],X)-matmult(self.A[:,j],Y[:,k]) )
                else:
                    Z[:,j,k] = solve(self.G, other.B[:,j,k]-matmult(self.B[:,j,k],X)-matmult(self.A[:,j],Y[:,k]) )
        
        return utb_derivative_system(X,Y,Z,2)