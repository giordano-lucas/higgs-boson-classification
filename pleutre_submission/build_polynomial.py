# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
from Scaler import *

# ===========================================================================
# ======================== PolynomialExpansion ==============================
# ===========================================================================

class PolynomialExpansion:
    """
    Object that performs polynomial feature 
    expansion and normalisation.
    """
    def __init__(self,degree,with_scaler=True,with_interractions=True):
        """
        Construct the PolynomialExpansion object
        @param degree      : (int) degree of the feature expansion
        @param with_scaler : (bool) True if the data should be normalised 
        @param with_interractions : (bool) True if interaction terms of 
        degree 2 should be included
        """
        self.scaler = None
        self.with_scaler = with_scaler
        self.degree = degree
        self.with_interractions=with_interractions
    
    def scale(self,X):
        """
        (private function)
        scales X if needed
        """
        if (self.scaler == None):           # in this case, a scaler is initialised
            self.scaler = StandardScaler()  # fit the scaler
            return self.scaler.fit(X)
        else:
            return self.scaler.transform(X) # otherwise we use the existing scaler

    def expand(self,X):
        """
        @param X       : (numpy array) input data
        @param degree  : (int) degree of the feature expansion
        @return X_poly : (numpy array) the expanded version of X
        """
        X_poly = build_poly(X,self.degree)    # add non interaction terms
        if self.with_interractions:
            interractions=get_interactions(X) # interractions of X
            X_poly= np.concatenate((X_poly, interractions), axis=1) # add the interraction terms to the final result
        if self.with_scaler:                  # if scaling is needed
            X_poly = self.scale(X_poly)
        X_poly = add_bias(X_poly)             # add a bias column to X_poly  
        return X_poly

# ===========================================================================
# ===========================================================================
# ===========================================================================

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if np.ndim(x)==1:           # handle case if X is one dimensional
        x = x[:,np.newaxis]     # add second dimension
    out  = x                    # output
    prev = out.copy()           # equat to X^{k+1} at iteration k 
    for d in range(degree-1):   # iterate over the degrees
        prev = prev * x         # construct X^{k+1} = X^{k} * X
        out  = np.concatenate((out, prev), axis=1) # concatenate
    return out                  # return the output

def add_bias(x):
    """ Add a column of ones to x (numpy array)"""
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