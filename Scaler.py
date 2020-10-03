

# ============================================================
# ================== Abstract Scaler =========================
# ============================================================
from abc import ABC, abstractmethod
import numpy as np

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

# ============================================================
# ===================  Sub Classes ===========================
# ============================================================

class StandardScaler(Scaler):
    """
    Standard Scaler that returns a matrix with 0-mean columns 
    and 1-std columns.
    """
    def fit(self,x):
        self.mean = np.mean(x,axis=0)
        self.std  = np.std(x,axis=0)
        return self.transform(x)

    def transform(self, x):
        return (x-self.mean)/self.std

class MinMaxScaler(Scaler):
    """
    Standard Scaler that returns a matrix scaler according to
    the formula : (x-min)/(max-min)
    """
    def fit(self,x):
        self.min = np.min(x,axis=0)
        self.max = np.max(x,axis=0)
        return self.transform(x)

    def transform(self, x):
        return (x-self.min)/(self.max-self.min)
