from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from scipy import stats
import statsmodels.api as sm
import numpy as np



class Regressor():
    def __init__(self):
        self.parameters = None
        self.regressor = None
        
    def Fit(self,X_train,Y_train):
        self.regressor.fit(X_train,Y_train,1)
        self.sse = np.sum((self.regressor.predict(X_train) - Y_train) ** 2, axis=0) / float(X_train.shape[0] - X_train.shape[1])
        self.se = np.array([
            np.sqrt(np.diagonal(self.sse[i] * np.linalg.inv(np.dot(X_train.T, X_train))))
                                                    for i in range(self.sse.shape[0])
                    ])
        self.t = self.regressor.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), Y_train.shape[0] - X_train.shape[1]))
        
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score
    
    def Predict(self,X_test):
        prediction = self.regressor.predict(X_test)
        return prediction
        

class OrdianryLeastSquareRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = LinearRegression()
      
class PartialLeastSqaureRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = PLSRegression()
    def Fit(self,X_train,Y_train,n_components):
        self.regressor.__dict__['n_components'] = n_components
        self.regressor.fit(X_train,Y_train)
        return self.regressor.get_params(), self.regressor.score
    
    
class LassoRegressor(Regressor):
    def __init__(self,alpha):
        super().__init__()
        self.regressor = Lasso(alpha)
    
    
class PrincipalComponentRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = OrdianryLeastSquareRegressor()
    def Fit(self,X_train,Y_train,n_components):
        self.pca = PCA(n_components)
        X_train_transform = self.pca.fit_transform(X_train)
        self.regressor.Fit(X_train_transform,Y_train)
    def Predict(self,X_test):
        X_test_transform = self.pca.transform(X_test)
        prediction = self.regressor.Predict(X_test_transform)
        return prediction
    
class RidgeRregressor(Regressor):
    def __init__(self,alpha):
        super().__init__()
        self.regressor = Ridge(alpha)