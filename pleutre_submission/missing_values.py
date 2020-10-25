from matplotlib.pyplot import axis
import numpy as np
from implementations import least_squares
from build_polynomial import add_bias
# ============================================================
# ============== Abstract Missing Handler ====================
# ============================================================
from abc import ABC, abstractmethod

class Interpolator(ABC):
    """
    Abstract class that defines an interpolator for
    the missing values.
    """
    @staticmethod
    def isnan(X,NAN_ENCODING=-999):
        return X<=NAN_ENCODING
    @staticmethod
    def to_nan(X):
        X[Interpolator.isnan(X)] = np.nan
        return X

    @abstractmethod
    def interpolate(self,X):
        pass

# ============================================================
# ===================  Sub Classes ===========================
# ============================================================
class MeanInterpolator(Interpolator):
    """
    Interpolator for missing values:
    
    For each feature, the missing values are interpolated to 
    the mean of the feature. 
    """
    def __init__(self):
        self.means = None
    def interpolate(self,X):
        """Set all the nan values to the mean of the corresponding feature"""
        # convert missing values into nan to be able to use numpys helper functions
        X = Interpolator.to_nan(X)
        if self.means is None:
            #feature means without the nans
            self.means=np.nanmean(X,axis=0)
        #Find indices that you need to replace
        inds = np.where(np.isnan(X))
        #Place column means in the indices. Align the arrays using take
        X[inds] = np.take(self.means, inds[1])
        return X

class LinearInterpolator(Interpolator):
    """
    Interpolator for missing values:
    
    For each feature 'c' that contains missing value the following procedure is 
    applied :
        - data is a mean-interpolated version of the dataset (named array). This
        is used to augment the training set size.
        - The column 'c' is used as the dependent variable in the least squares
        setting and the remaining ones define the independent variables.
        - data is splitted into a training/test set : all the missing values go 
        into the test set and the remaining ones into the training set. The
        idea is to use the existing values to predict the missing ones.
        - a simple least squares model is fitted on the training set
        - missing values are predict based on the model. 
    Note that the weights are only computed the first time that the method is 
    called, after that the prediction are based on the previously computed weights.
    """
    def __init__(self):
        self.weights  = None
        self.nan_cols = None
        self.train    = True
    def interpolate(self,array):
        # convert missing values into nan to be able to use numpys helper functions
        data = MeanInterpolator().interpolate(np.array(array,copy=True))
        array=Interpolator.to_nan(array)
        # get the columns where is there at least one nan
        if self.train: # initialise object
            self.nan_cols = np.where(np.all(~np.isnan(data),axis=0))[0]
            self.weights  = np.zeros((len(self.nan_cols),data.shape[1]))
        # for each columns, train a least square interpolation
        for i,c in enumerate(self.nan_cols): 
            train_mask = ~np.isnan(array[:,c]) # filter nans 
            if self.train:
                y = data[train_mask,c]
                X = data[train_mask,:][:,np.arange(data.shape[1])!=c]
                tX = add_bias(X)                            # add bias
                self.weights[i],loss = least_squares(y,tX)  # train
            # *** prediction ***
            pred_mask = ~train_mask
            X_pred  = data[pred_mask,:][:,np.arange(data.shape[1])!=c]
            tX_pred = add_bias(X_pred) 
            y_pred  = tX_pred @ self.weights[i]
            array[pred_mask,c] = y_pred
        self.train = False
        return array