import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression

class Regressor():
    def __init__(self):
        self.parameters = None
        self.regressor = None
        
    def Fit(self):
        raise NotImplementedError
    
    def Predict(self):
        raise NotImplementedError
        

class OrdianryLeastSquareRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = LinearRegression()
        
    def Fit(self,X_train,Y_train):
        self.regressor.fit(X_train,Y_train)
        # this function would return the parameters of the regressor and the R^2.
        return self.regressor.get_params(), self.regressor.score 
    def Predict(self,X_test):
        prediction = self.regressor.predict(X_test)
        return prediction
      
class PartialLeastSqaureRegressor(Regressor):
    def __init__(self):
        super().__init__()
    
    def Fit(self,X_train,Y_train,n_components):
        pass
    
    def Predict(self,X_test):
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