from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from scipy import stats
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


class Regressor():
    def __init__(self):
        self.parameters = None
        self.regressor = None
    
    def _inference(self,X_train,Y_train):
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
        scatter = plt.scatter(X,Y)
        line = plt.plot(X,self.regressor.predict(X))
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
        X_train_transform = self.pca.fit_transform(X_train)
        self.regressor.fit(X_train_transform,Y_train)
        self._inference(X_train,Y_train)
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(X_train,Y_train)
        
    def Predict(self,X_test):
        X_test_transform = self.pca.transform(X_test)
        prediction = self.regressor.Predict(X_test_transform)
        return prediction
    
class RidgeRregressor(Regressor):
    def __init__(self,alpha):
        super().__init__()
        self.regressor = Ridge(alpha)