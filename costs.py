import numpy as np
# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    return mse(y,tx,w)

def mse(y,tx,w):
    return 1/2 * np.mean((y-tx@w)**2)
def mae(y,tx,w):
    return np.mean(np.abs((y-tx@w))) 