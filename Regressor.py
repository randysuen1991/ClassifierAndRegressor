from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression, Lars
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sliced import SlicedInverseRegression

"""
Notice: I should add , PIRE(partial inverse regression), decision tree, ...regressions to this file.

"""


class Regressor():
    def __init__(self):
        self.parameters = None
        self.regressor = None
    
    def _inference(self,X_train,Y_train):
        if type(X_train) == pd.DataFrame :
            X_train == X_train.values
            
            
        if type(self.regressor) == Lasso :
            predictions = np.expand_dims(self.regressor.predict(X_train),1)
            self.sse = np.sum(( predictions - Y_train) ** 2, axis=0) / float(X_train.shape[0] - X_train.shape[1])
        else :
            self.sse = np.sum((self.regressor.predict(X_train) - Y_train) ** 2, axis=0) / float(X_train.shape[0] - X_train.shape[1])
            
            
        if type(self.sse) == np.float64 :
            self.sse = [self.sse]
        self.se = np.array([
            np.sqrt(np.diagonal(self.sse[i] * np.linalg.inv(np.dot(X_train.T, X_train))))
                                                    for i in range(len(self.sse))
                    ])        
        #self.se = beta_i_hat / sqrt(std_sqr multiplied inv((X.T)X)_ii ) 
        self.t = self.regressor.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), X_train.shape[0] - X_train.shape[1]))
    
    
    def Fit(self,X_train,Y_train):
        self.regressor.fit(X_train,Y_train)
        self._inference(X_train,Y_train)
        
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(X_train,Y_train)
    
    def Predict(self,X_test):
        prediction = self.regressor.predict(X_test)
        return prediction
    
    def RegressionPlot(self,X,Y):
        scatter = plt.scatter(X,Y,color='b')
        line = plt.plot(X,self.regressor.predict(X),color='r')
        plt.ylabel('response')
        plt.xlabel('explanatory')
        plt.legend(handles=[scatter,line[0],],labels=['Scatter Plot','Intercept:{}, Slope:{},\n R-square:{}'.format(self.regressor.intercept_,self.regressor.coef_,self.regressor.score(X,Y))],loc='best')
        plt.title('Scatter Plot and Regression')

class OrdianryLeastSquareRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = LinearRegression()
      
class PartialLeastSqaureRegressor(Regressor):
    def __init__(self,n_components):
        super().__init__()
        self.regressor = PLSRegression(n_components=n_components)    
    def Fit(self,X_train,Y_train):
        self.regressor.fit(X_train,Y_train)
        self._inference(X_train,Y_train)
        
        return None, self.regressor.coef_, self.p, self.regressor.score(X_train,Y_train)
    
class LassoRegressor(Regressor):
    def __init__(self,alpha):
        super().__init__()
        self.regressor = Lasso(alpha)
    
# Still need to check this class.
class PrincipalComponentRegressor(Regressor):
    def __init__(self,n_components):
        super().__init__()
        self.n_components = n_components
        self.regressor = LinearRegression()
    def Fit(self,X_train,Y_train):
        self.pca = PCA(self.n_components)
        self.X_train_transform = self.pca.fit_transform(X_train)
        self.regressor.fit(self.X_train_transform,Y_train)
        self._inference(self.X_train_transform,Y_train)
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(self.X_train_transform,Y_train) 
    def Predict(self,X_test):
        X_test_transform = self.pca.transform(X_test)
        prediction = self.regressor.Predict(X_test_transform)
        return prediction
    
class RidgeRegressor(Regressor):
    def __init__(self,alpha):
        super().__init__()
        self.regressor = Ridge(alpha)
        
        
class RandForestRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = RandomForestRegressor()
        
        
class SlicedInverseRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = SlicedInverseRegression()
        
class LeastAngleRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = Lars()
        