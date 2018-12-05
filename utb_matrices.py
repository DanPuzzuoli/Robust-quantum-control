#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:09:36 2018

@author: Daniel Puzzuoli

Description: 
    This file contains various functions for constructing and playing with
    upper triangular block matrices, as well as computing certain quantities using 
    upper triangular block matrices. 

Notes:
    At the moment this is just a place to throw some related functions
"""

from numpy import block,zeros
from scipy.linalg import expm

#do this next
#def exp_second_deriv(A,B1,B2):

def exp_deriv(A,B):
    """
    This function computes the derivative of
    exp(A+eB) w.r.t. e at e = 0, using upper triangular block matrices. 
    As exp(A) is also computed as a byproduct, it also returns exp(A) 
    
    Parameters
    ----------
    A : numpy.array
        The matrix to be exponentiated
    B : numpy.array
        The direction of the derivative for exp(A+eB)

    Returns
    -------
    two numpy arrays in a tuple
        exp(A), derivative of exp(A+eB) w.r.t. e at e=0
    """
    d = len(A)
    m = expm(decoupling_gen(A,[B]))
    return get_block(m, d, 0, 0),get_block(m, d, 0, 1)

def decoupling_gen(G,Alist):
    """
    Constructs a block upper triangular matrix with block size len(Alist) + 1,
    with every entry on the diagonal being G, and the first off diagonal
    consisting of entries of Alist
    
    E.g. G,[X,Y] -> 
                block( [[G, X, 0],
                        [0, G, Y],
                        [0, 0, G]])
    
    Parameters
    ----------
    G : 2d numpy array
        The matrix to appear in the diagonal entries
    Alist : list of 2d numpy arrays, or 2d numpy array
        The matrices to appear in the first off-diagonal, assumed to have
        the same dimension as G.

    Returns
    -------
    numpy array
        the desired block upper triangular matrix, as given in the description
    """
    
    if type(Alist) is not list:
        return decoupling_gen(G,[Alist])
    else:
        return utb_from_diag([[G]*(len(Alist)+1), Alist])


def utb_from_diag(diags):
    """
    Constructs a upper triangular block matrix out of a specification in terms
    of the diagonals.
    
    E.g. [[X,Y], [Z]] -> numpy.block([[X,Z],[0,Y]])

    Parameters
    ----------
    diags :
        A list of lists of numpy arrays, with each array assumed to be the same
        square dimension, with each list representing a successive diagonal in
        a block upper triangular matrix. Each list is assumed to have length 1
        less than the last, and the last is assumed to have length 1.

    Returns
    -------
    numpy array
        A block upper triangular matrix, with the entries of each diagonal as
        specified by the input

    """
    # get zero matrix of appropriate size
    m0 = zeros(diags[0][0].shape)
    
    # complete the list of diagonals by appending lists of 0 matrices of
    # length decreasing by 1 until the last entry of diags is length 1
    while len(diags[-1]) > 1:
        diags.append( [m0]* ( len(diags[-1])-1 ) )
    
    return utb_from_rows(triangle_list_transpose(diags))


def utb_from_rows(rows):
    """
    Constructs an upper triangular block matrix from a specification of its
    rows.
    
    E.g. for X,Y,Z square arrays of the same dimension, maps [[X,Y], [Z]] to
    numpy.block([[X,Y], [0, Z]]), where 0 is the zero array of the appropriate
    dimension
    
    Parameters
    ----------
    rows :
        a list of lists containing numpy arrays representing the blocks in each
        row, starting from the diagonal. All arrays are assumed to be square 
        and of the same dimension. Each row is assumed to have length 1 less
        than the last, with the last being length 1.

    Returns
    -------
    numpy array
        A block upper triangular matrix, with the entries of each row above
        the sub-diagonal being those specified by the input
    """
    #define the zero matrix of the appropriate shape
    m0 = zeros(rows[0][0].shape)
    
    
    # prepend each row with the appropriate number of zero matrices, so that
    # all rows have the same length
    for i in range(0, len(rows[0])):
        for k in range(0, i):
            rows[i].insert(0,m0)
    
    return block(rows)



def triangle_list_transpose(triangle):
    """ 
    Performs transposition on a triangular list. The purpose here is to
    transform the specification of an upper triangular block matrix in terms of
    the rows into one in terms of the diagonals, and vice versa
    
    Input: 
        a list of lists with `triangular shape', i.e. each successive list 
        has length one less than the previous list, and the last list has
        length 1
    Output:
        the transpose of the list
        e.g. [['a', 'b'], ['c']] -> [['a', 'c'], ['b']]
    """
    rows = []
    for i in range(0, len(triangle)):
        rows.append([x[i] for x in triangle[0: len(triangle)-i]])
    
    return rows
            

def get_block(block_matrix, block_dim, row, col):
    """ 
    Given a block matrix, the dimension of the blocks, and a specified location
    of a block, returns the block. indexing of block locations is assumed
    to start at 0. 
    
    Input: 
    block_matrix : numpy array
        a block matrix with block size block_dim
    block_dim : int
        the dimension of each block
    row : int
        The row of the desired block
    col : int
        The column of the desired block
        
    Output:
    numpy array
        the specified block
        
    E.g. given the numpy array with structure [[A,B], [C,D]], where A,B,C,D are
    are square matrices of size d, get_block([[A,B], [C,D]], d, 0, 1) = B
    """
    # the rows to extract
    rstart = row*block_dim
    rend = (row+1)*block_dim
    
    #the columns to extract
    cstart = col*block_dim
    cend = (col+1)*block_dim
    return block_matrix[rstart:rend, cstart:cend]