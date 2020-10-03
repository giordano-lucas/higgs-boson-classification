

# ============================================================
# ================== Abstract Scaler =========================
# ============================================================
from abc import ABC, abstractmethod
import numpy as np

class Scaler(ABC):
    @abstractmethod
    def fit(self,x):
        pass
    @abstractmethod
    def transform(self,x):
        pass

# ============================================================
# ===================  Sub Classes ===========================
# ============================================================

class StandardScaler(Scaler):
    def fit(self,x):
        self.mean = np.mean(x,axis=0)
        self.std  = np.std(x,axis=0)
        return self.transform(x)

    def transform(self, x):
        return (x-self.mean)/self.std

class MinMaxScaler(Scaler):
    def fit(self,x):
        self.min = np.min(x,axis=0)
        self.max = np.max(x,axis=0)
        return self.transform(x)

    def transform(self, x):
        return (x-self.min)/(self.max-self.min)
