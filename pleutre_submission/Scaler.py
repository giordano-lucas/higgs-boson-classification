

# ============================================================
# ================== Abstract Scaler =========================
# ============================================================
from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.shape_base import hsplit

class Scaler(ABC):
    """
    Abstract class that definies a scaler : a function that 
    'normalises' the data according to some function.
    """
    @abstractmethod
    def fit(self,x):
        """
        Function that computes the paramters of the normalisa-
        tion process and stores them into the instance of the
        object. 
        It is then supposed to call the method : transform and
        return the result.

        @params x : (numpy array)
        @return x_normalised : after after applying the norma-
        lisation process 
        """
        pass
    @abstractmethod
    def transform(self,x):
        """
        Performs the actual normalisation with the help of the
        parameters defined by the previous call to : fit

        /!\ fit should be called before transform /!\
        
        @params x : (numpy array)
        @return x_normalised : after after applying the norma-
        lisation process 
        """
        pass

    def __init__(self,has_bias=False):
        self.correct_bias = (1 if has_bias else 0)
# ============================================================
# ===================  Sub Classes ===========================
# ============================================================

class StandardScaler(Scaler):
    """
    Standard Scaler that returns a matrix with 0-mean columns 
    and 1-std columns.
    """
   
    def fit(self,x):
        self.mean = np.mean(x[:,self.correct_bias:],axis=0)
        self.std  = np.std(x[:,self.correct_bias:],axis=0)
        return self.transform(x)

    def transform(self, x):
        x[:,self.correct_bias:] = (x[:,self.correct_bias:]-self.mean)/self.std
        return x

class MinMaxScaler(Scaler):
    """
    Standard Scaler that returns a matrix scaler according to
    the formula : (x-min)/(max-min)
    """
    def fit(self,x):
        x_t = self.correct_bias(x)
        self.min = np.min(x,axis=0)
        self.max = np.max(x,axis=0)
        return self.transform(x)

    def transform(self, x):
        x_t = self.correct_bias(x)
        x_t = self.correct_bias(x)
        x_t = (x_t-self.min)/(self.max-self.min)
        return x #since x_t is a view on x
