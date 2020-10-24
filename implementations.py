# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
from proj1_helpers import *

#================================================================
#=============== General Hepler Functions =======================
#================================================================

def gradient_descent(y, tx, initial_w,max_iters, gamma,compute_loss,compute_gradient):
    """
    Helper function that performs the gradient descent algorithm
    Extra parameters :
        @param compute_loss: (function) computes the value of the loss
        function associated to the GD algorithm. 
        Should take the following list of arguments : (y,tx,w)
        @param compute_gradient: (function) computes the gradient of w. 
        Should take the following list of arguments : (y,tx,w)
    """
    w=initial_w
    for n_iter in range(max_iters):
        
        grd=compute_gradient(y,tx,w)
        w=w-gamma*grd
        loss=compute_loss(y,tx,w)
        if n_iter % 500 == 0:
            print("Current iteration = {i}, loss={l}".format(i=n_iter, l=loss))
    loss=compute_loss(y,tx,w)
    return w,loss

#================================================================
#====================== Least Squares ===========================
#================================================================

def least_squares_GD(y, tx, initial_w,max_iters, gamma):
    """Least squares with Gradient descent algorithm"""
    return gradient_descent(
        y, tx, initial_w,max_iters, gamma,
        loss_least_squares,
        gradient_least_squares)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Least squares with Stochastic Gradient descent algorithm"""
    return gradient_descent(
        y, tx, initial_w,max_iters, gamma,
        loss_least_squares,
        stoch_gradient_least_squares)

def least_squares(y, tx):
    """calculate the least squares solution."""
    # returns optimal weights, and mse
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    return w, loss_least_squares(y,tx,w)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.linalg.solve(tx.T@tx + lambda_*np.eye(tx.shape[1]),tx.T@y)
    return w, loss_least_squares(y,tx,w)

#===================== Hepler functions =========================

def loss_least_squares(y, tx, w):
    """ Compute the loss using mse """
    e = y-tx@w
    return 1/(2*len(y))*(e@e)

def gradient_least_squares(y, tx, w):
    """Compute the gradient of mse """
    e = y-tx@w
    return -(1/len(y)) * tx.T @ e

def stoch_gradient_least_squares(y, tx, w):
    """
    Compute a stochastic gradient from just few examples 
    n and their corresponding y_n labels.
    """   
    by,bx = [b for b in batch_iter(y, tx, batch_size=1)][0]
    return gradient_least_squares(by, bx, w)

#================================================================
#================== Logistic Regression =========================
#================================================================

def logistic_regression(y,tx,initial_w,max_iters,gamma):
    """Logistic regression with gradient descent"""
    return gradient_descent(
        y, tx, initial_w,max_iters, gamma,
        loss_logistic_regression,
        gradient_loss_logistic_regression)

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """Logistic regression with stochastic gradient descent"""
    return gradient_descent(
        y, tx, initial_w,max_iters, gamma,
        lambda y,tx,w: loss_reg_logistic_regression(y,tx,w,lambda_),
        lambda y,tx,w: gradient_reg_logistic_regression(y,tx,w,lambda_))

#===================== Hepler functions =========================

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1+np.exp(-t))

def loss_logistic_regression(y, tx, w):
    """compute the loss: negative log likelihood."""
    z = tx@w
    return np.sum(np.log(1+np.exp(z))-y*z)

def gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w) - y)

def loss_reg_logistic_regression(y, tx, w,lambda_):
    """compute the loss: negative log likelihood."""
    norm = (1/2)*lambda_*np.sum(w**2) # norm of w 
    return loss_logistic_regression(y,tx,w)+norm

def gradient_reg_logistic_regression(y, tx, w,lambda_):
    """compute the gradient of loss."""
    grad_norm = lambda_*w # gradient of the normal w
    return gradient_logistic_regression(y,tx,w)+grad_norm