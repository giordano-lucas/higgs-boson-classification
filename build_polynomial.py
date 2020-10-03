# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
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
    ones = np.ones((x.shape[0],1))
    return np.concatenate((ones,x), axis=1)