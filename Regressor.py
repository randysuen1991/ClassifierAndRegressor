from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression

# import statsmodels.api as sm



class Regressor():
    def __init__(self):
        self.parameters = None
        self.regressor = None
        
    def Fit(self,X_train,Y_train):
        self.regressor.fit(X_train,Y_train)
        # this function would return the parameters of the regressor and the R^2.
        return self.regressor.intercept_[0], self.regressor.coef_, self.regressor.score
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