# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
from Scaler import *

# ===========================================================================
# ===========================================================================
# ===========================================================================

class PolynomialExpansion:
    """
    Object that performs polynomial feature 
    expansion and normalisation.
    """
    def __init__(self,degree,with_scaler=True,with_interractions=False):
        self.scaler = None
        self.with_scaler = with_scaler
        self.degree = degree
        self.with_interractions=with_interractions
    
    def scale(self,X):
        if (self.scaler == None):           # in this case, a scaler is initialised
            self.scaler = StandardScaler()  # fit the scaler
            return self.scaler.fit(X)
        else:
            return self.scaler.transform(X) # otherwise we use the existing scaler

    def expand(self,X):
        """
        @param X : numpy array
        @param degree : degree of the feature 
        expansion
        @return X_poly : the expanded version
        of X
        """
        X_poly = build_poly(X,self.degree)    # add non interaction terms
        
        
        if self.with_interractions:
            interractions=get_interactions(X) #interractions of X
            X_poly= np.concatenate((X_poly, interractions), axis=1) # add the interraction terms to the final result
            
        if self.with_scaler:                  # if scaling is needed
            X_poly = self.scale(X_poly)
        X_poly = add_bias(X_poly)             # add a bias column to X_poly  
       
        return X_poly

# ===========================================================================
# ===========================================================================
# ===========================================================================

def build_poly(x, degree,add_interractions=False):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    if np.ndim(x)==1:
        x = x[:,np.newaxis]
        
    out  = x
    prev = out.copy()
    for d in range(degree-1):
        prev = prev * x
        out  = np.concatenate((out, prev), axis=1)
  
    # ***************************************************
    return out

def add_bias(x):
    """
    Add a column of ones to x 
    """
    ones = np.ones((x.shape[0],1))
    return np.concatenate((ones,x), axis=1)


def get_interactions(x):
    """
    expand the features (columns) of a given matrix x
    by all the interaction terms of degree 2
    """
    #array containing all create features plus the old ones 
    interractions=[]
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i>j: #condition to avoid duplicate (xy=yx)
                interractions.append(x[:,i]*x[:,j])
    return np.column_stack(interractions)