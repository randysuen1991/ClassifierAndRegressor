import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge

class Regressor():
    def __init__(self):
        self.parameters = None
        
    def Fit(self):
        raise NotImplementedError
    
    def Predict(self):
        raise NotImplementedError
        
        
class PartialLeastSqaureRegressor(Regressor):
    def __init__(self):
        super().__init__()
    
    def Fit(self,X_train,Y_train,n_components):
        pass
    
    def Predict(self):
        pass
    
class LassoRegressor(Regressor):
    def __init__(self):
        super().__init__()
        
    def Fit(self,X_train,Y_train,n_components):
        pass
    
    def Predict(self):
        pass
    
    
class PrincipalComponentRegressor(Regressor):
    def __init__(self):
        super().__init__()
        
    def Fit(self,X_train,Y_train,n_components):
        pass
    
    def Predict(self):
        pass
    
class RidgeRregressor(Regressor):
    def __init__(self):
        super().__init__()
        
    def Fit(self,X_train,Y_train,n_components):
        pass
    
    def Predict(self):
        pass