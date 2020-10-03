# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
from proj1_helpers import *


#Gradient descent algorithm
def least_squares_GD(y, tx, initial_w,max_iters, gamma):

    w=initial_w
    
    for n_iter in range(max_iters):
        grd=compute_gradient(y,tx,w)
        w=w-gamma*grd
        
        loss=compute_loss(y,tx,w)
        if n_iter%10==0:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    loss=compute_loss(y,tx,w)
    return w,loss

#Compute the loss using mse
def compute_loss(y, tx, w):
    e=y-tx.dot(w)
    n=len(y)
    return 1/(2*n)*e.dot(e)

#Compute the gradient of the loss function (mse)
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=y-tx.dot(w)
    grd=tx.T.dot(e)
    return -(1/len(y))*grd 

def compute_stoch_least_squares_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""   
    my=None
    mx=None
    for mmini_y,mini_x in batch_iter(y, tx, batch_size=1):
            my=mmini_y
            mx=mini_x
    
    return compute_gradient(my, mx, w)


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    "Gradient descent algorithm"
    w=initial_w
    for i in range(max_iters):
        grd=gamma*compute_stoch_least_squares_gradient(y,tx,w)
        w=w-grd
    loss=compute_loss(y,tx,w)
    
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""

    # returns optimal weights, and mse
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    return w, compute_loss(y,tx,w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    w = np.linalg.solve(tx.T@tx + lambda_*np.eye(tx.shape[1]),tx.T@y)
    return w, compute_loss(y,tx,w)
