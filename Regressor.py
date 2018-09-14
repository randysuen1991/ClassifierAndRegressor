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
import ModelEvaluation as ME
import ModelSelection as MS
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import ResidualsPlot
import yellowbrick


import sys
if 'C:\\Users\\randysuen\\pycodes\\Dimension-Reduction-Approaches' not in sys.path:
    sys.path.append('C:\\Users\\randysuen\\pycodes\\Dimension-Reduction-Approaches')
if '/home/randysuen/pycodes/Dimension-Reduction-Approaches' not in sys.path:
    sys.path.append('/home/randysuen/pycodes/Dimension-Reduction-Approaches')
from DimensionReductionApproaches import CenteringDecorator, StandardizingDecorator

"""
Notice: I should add , PIRE(partial inverse regression), decision tree, ...regressions to this file.
"""

# There should be stagewise.


class Regressor:
    def __init__(self):
        self.parameters = dict()
        self.regressor = None
        self.sse = None
        self.sst = None
        self.adjrsquared = None
        self.rsquared = None
        self._X_train = None
        self._Y_train = None
        self.n = None
        # x_k refers to the number of the predictors of X_train
        self.x_k = None
        # y_k refers to the number of the responses of Y_train
        self.y_k = None
        self.p = None
        self.standardize = None
        self.standardizescaler = StandardScaler()
        self.residual_visualizer = None

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, X_train):
        self._X_train = X_train
        try:
            self.x_k = X_train.shape[1]
            self.n = X_train.shape[0]
        except IndexError:
            self.x_k = 1
            self.n = X_train.shape[0]
            self._X_train = self._X_train.reshape(-1, 1)

    @property
    def Y_train(self):
        return self._Y_train

    @Y_train.setter
    def Y_train(self, Y_train):
        self._Y_train = Y_train
        try:
            self.y_k = Y_train.shape[1]
        except IndexError:
            self.y_k = Y_train.shape[0]
            self._Y_train = self._Y_train.reshape(-1, 1)

    def _Inference(self):

        try:
            self.rsquared = self.regressor.score(self.X_train, self.Y_train)
        except AttributeError:
            self.rsquared = self.regressor.regressor.score(self.X_train, self.Y_train)

        self.adjrsquared = ME.ModelEvaluation.AdjRsquared(self)

        # Store some info of the model.
        self.sst = np.sum((self.Y_train-np.mean(self.Y_train, axis=0))**2, axis=0)

        if type(self.regressor) == Lasso:
            predictions = np.expand_dims(self.Predict(self.X_train), 1)
            self.sse = np.sum((predictions - self.Y_train) ** 2, axis=0)
        else:
            self.sse = np.sum((self.Predict(self.X_train) - self.Y_train) ** 2, axis=0)
        self.sse_scaled = self.sse / float(self.X_train.shape[0] - self.X_train.shape[1])

        if type(self.sse_scaled) == np.float64:
            self.sse_scaled = [self.sse_scaled]

        try:
            centered_X_train = self.X_train - np.mean(self.X_train, axis=0)
            self.se = np.array([np.sqrt(np.diagonal(self.sse_scaled[i] *
                                                    np.linalg.inv(np.dot(centered_X_train.T, centered_X_train))))
                                for i in range(len(self.sse_scaled))])
        except np.linalg.linalg.LinAlgError:
            return
        except TypeError:
            return
        try:
            self.t = self.regressor.coef_ / self.se
        except AttributeError:
            try:
                self.t = self.parameters['beta'] / self.se
            except KeyError:
                return
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), self.X_train.shape[0] - self.X_train.shape[1]))

    def Fit(self, X_train, Y_train, standardize=False):
        self.standardize = standardize
        if self.standardize:
            self.standardizescaler.fit(X_train)
            X_train = self.standardizescaler.transform(X_train)
        plt.plot(X_train, Y_train)
        plt.show()
        self.X_train = X_train
        self.Y_train = Y_train
        self.regressor.fit(self.X_train, self.Y_train)
        self._Inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(X_train, Y_train)

    def Predict(self, X_test):
        if self.standardize:
            X_test = self.standardizescaler.transform(X_test)
        try:
            return self.regressor.predict(X_test)
        except AttributeError:
            return self.regressor.Predict(X_test=X_test)


    def Regression_Plot(self, X_test, Y_test):
        scatter = plt.scatter(X_test, Y_test, color='b')
        try:
            line = plt.plot(X_test, self.regressor.predict(X_test), color='r')
        except AttributeError:
            line = plt.plot(X_test, self.regressor.Predict(X_test), color='r')
        plt.ylabel('response')
        plt.xlabel('explanatory')
        plt.legend(handles=[scatter, line[0], ],
                   labels=['Scatter Plot',
                           'Intercept:{}, Slope:{},\n R-square:{}'.format(self.regressor.intercept_,
                                                                          self.regressor.coef_,
                                                                          self.regressor.score(X_test, Y_test))],
                   loc='best')
        plt.title('Scatter Plot and Regression')

    def Residual_Plot(self, X_test=None, Y_test=None):
        if self.standardize:
            X_test = self.standardizescaler.transform(X_test)
        try:
            self.residual_visualizer = ResidualsPlot(self.regressor)
        except yellowbrick.exceptions.YellowbrickTypeError:
            self.residual_visualizer = ResidualsPlot(self.regressor.regressor)

        self.residual_visualizer.fit(self.X_train, self.Y_train)
        if X_test is not None and Y_test is not None:
            self.residual_visualizer.score(X_test, Y_test)
        self.residual_visualizer.poof()

    def Get_Score(self, X_test, Y_test):
        if self.standardize:
            X_test = self.standardizescaler.transform(X_test)
        try:
            return self.regressor.score(X_test, Y_test)
        except AttributeError:
            return self.regressor.Get_Score(X_test, Y_test)


class OrdinaryLeastSquaredRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = LinearRegression()


class PartialLeastSqaureRegressor(Regressor):
    def __init__(self, n_components):
        super().__init__()
        self.regressor = PLSRegression(n_components=n_components)

    def Fit(self, X_train, Y_train):
        self.regressor.fit(X_train, Y_train)
        self.Y_train = Y_train
        self.X_train = X_train
        self._Inference()
        
        return None, self.regressor.coef_, self.p, self.regressor.score(X_train, Y_train)


class LassoRegressor(Regressor):
    def __init__(self, alpha):
        super().__init__()
        self.regressor = Lasso(alpha)


# Still need to check this class.
class PrincipalComponentRegressor(Regressor):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.regressor = LinearRegression()
        self.pca = None

    def Fit(self, X_train, Y_train, standardize=False):
        self.pca = PCA(self.n_components)
        self.X_train = self.pca.fit_transform(X_train)
        self.Y_train = Y_train
        self.regressor.fit(self.X_train, self.Y_train)
        self._Inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(self.X_train, Y_train)

    def Predict(self, X_test):
        try:
            X_test_transform = self.pca.transform(X_test)
        except ValueError:
            X_test_transform = X_test
        try:
            prediction = self.regressor.Predict(X_test_transform)
        except AttributeError:
            prediction = self.regressor.predict(X_test_transform)
        return prediction

    def Residual_Plot(self, X_test=None, Y_test=None):
        if self.standardize:
            X_test = self.standardizescaler.transform(X_test)
        try:
            self.residual_visualizer = ResidualsPlot(self.regressor)
        except yellowbrick.exceptions.YellowbrickTypeError:
            self.residual_visualizer = ResidualsPlot(self.regressor.regressor)

        self.residual_visualizer.fit(self.X_train, self.Y_train)
        if X_test is not None and Y_test is not None:
            try:
                self.residual_visualizer.score(X_test, Y_test)
            except ValueError:
                X_test = self.pca.transform(X_test)
                self.residual_visualizer.score(X_test, Y_test)
        self.residual_visualizer.poof()


class RidgeRegressor(Regressor):
    def __init__(self, alpha):
        super().__init__()
        self.regressor = Ridge(alpha)
        
        
class RandForestRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = RandomForestRegressor()

    def Fit(self, X_train, Y_train, standardize=False):
        self.standardize = standardize
        if self.standardize:
            self.standardizescaler.fit(X_train)
            X_train = self.standardizescaler.transform(X_train)

        self.X_train = X_train
        self.Y_train = Y_train
        self.regressor.fit(self.X_train, self.Y_train.ravel())
        self._Inference()
        return self.rsquared

    def Residual_Plot(self, X_test=None, Y_test=None):
        if self.standardize:
            X_test = self.standardizescaler.transform(X_test)
        try:
            self.residual_visualizer = ResidualsPlot(self.regressor)
        except yellowbrick.exceptions.YellowbrickTypeError:
            self.residual_visualizer = ResidualsPlot(self.regressor.regressor)

        Y_train = self.Y_train.ravel()
        self.residual_visualizer.fit(self.X_train, Y_train)
        if X_test is not None and Y_test is not None:
            Y_test = Y_test.ravel()
            self.residual_visualizer.score(X_test, Y_test)
        self.residual_visualizer.poof()

    def Predict(self, X_test):
        if self.standardize:
            X_test = self.standardizescaler.transform(X_test)
        try:
            return self.regressor.predict(X_test).reshape(-1, 1)
        except AttributeError:
            return self.regressor.Predict(X_test=X_test).reshape(-1, 1)


class SlicedInverseRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = SlicedInverseRegression()


class LeastAngleRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = Lars()


class ForwardStepwiseRegressor(Regressor):
    def __init__(self, regressor=OrdinaryLeastSquaredRegressor, criteria=ME.ModelEvaluation.AIC):
        super().__init__()
        self.regressor_type = regressor
        self.regressor = regressor()
        self.criteria = criteria
        self.pred_ind = None

    def Fit(self, X_train, Y_train, standardize=False, **kwargs):
        self.standardize = standardize

        self.pred_ind = MS.ModelSelection.ForwardSelection(model=self.regressor_type, X_train=X_train,
                                                           Y_train=Y_train, p=kwargs.get('p', X_train.shape[1]),
                                                           standardize=self.standardize)
        if self.standardize:
            self.standardizescaler.fit(X_train[:, self.pred_ind])
            self.X_train = self.standardizescaler.transform(X_train[:, self.pred_ind])
        else:
            self.X_train = X_train[:, self.pred_ind]

        self.Y_train = Y_train

        try:
            self.regressor.fit(self.X_train, self.Y_train)
        except AttributeError:
            self.regressor.Fit(self.X_train, self.Y_train)

        self._Inference()
        try:
            return self.regressor.intercept_, self.regressor.coef_, self.p, \
                self.regressor.score(self.X_train, self.Y_train), self.pred_ind
        except AttributeError:
            return self.regressor.regressor.intercept_, self.regressor.regressor.coef_, self.p, \
                   self.regressor.regressor.score(self.X_train, self.Y_train), self.pred_ind


class BackwardStepwiseRegressor(Regressor):
    def __init__(self, regressor=OrdinaryLeastSquaredRegressor, criteria=ME.ModelEvaluation.AIC):
        super().__init__()
        self.regressor_type = regressor
        self.regressor = regressor()
        self.criteria = criteria
        self.pred_ind = None

    def Fit(self, X_train, Y_train, standardize=False, **kwargs):
        self.standardize = standardize
        self.pred_ind = MS.ModelSelection.BackwardSelection(model=self.regressor_type, X_train=X_train,
                                                            Y_train=Y_train, p=kwargs.get('p', X_train.shape[1]),
                                                            standardize=self.standardize)
        if self.standardize:
            self.X_train = self.standardizescaler.fit_transform(X_train[:, self.pred_ind])
        else:
            self.X_train = X_train[:, self.pred_ind]

        self.Y_train = Y_train
        self.regressor.fit(self.X_train, self.Y_train)
        self._Inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, \
            self.regressor.score(self.X_train, self.Y_train), self.pred_ind


class BestsubsetRegressor(Regressor):
    def __init__(self, criteria=ME.ModelEvaluation.AIC):
        super().__init__()
        self.regressor = LinearRegression()
        self.criteria = criteria

    def Fit(self, X_train, Y_train, **kwargs):
        ids = MS.ModelSelection.BestSubsetSelection(model=OrdinaryLeastSquaredRegressor, X_train=X_train,
                                                    Y_train=Y_train, p=kwargs.get('p', X_train.shape[1]))
        self.X_train = X_train[:, ids]
        self.Y_train = Y_train
        self.regressor.fit(self.X_train, self.Y_train)
        self._Inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, \
               self.regressor.score(self.X_train, self.Y_train), ids


# The response should be univariate.
class ForwardStagewiseRegressor(Regressor):

    def __init__(self):
        super().__init__()
        self.X_mean = None
        self.Y_mean = None

    @CenteringDecorator('X_train', 'Y_train')
    @StandardizingDecorator('X_train', 'Y_train')
    def Fit(self, X_train, Y_train, **kwargs):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_mean = np.mean(X_train, axis=0)
        self.Y_mean = np.mean(Y_train, axis=0)
        eps = kwargs.get('eps', 0.01)
        k = kwargs.get('k', self.x_k)
        lower_bound = kwargs.get('lower_bound', 0)
        assert k <= self.x_k
        residual = Y_train
        available_predictors = list(range(k))
        cors = np.zeros(shape=(k, ))
        beta = np.zeros(shape=(k, ))
        for i in range(k):
            for predictor in available_predictors:
                cors[predictor] = np.matmul(residual.T, X_train[:, predictor])
            abs_cors = [np.abs(cor) for cor in cors]
            index = np.argmax(abs_cors)
            if abs_cors[index] < lower_bound:
                break
            sign = np.sign(cors[index])
            beta[index] += sign * eps
            residual -= sign * eps * np.expand_dims(X_train[:, index], axis=1)

            available_predictors.remove(index)
            cors[index] = 0
            abs_cors[index] = 0

        self.parameters['beta'] = beta
        self._Inference(X_train, Y_train)

        return 0, self.parameters['beta'], self.p, self.rsquared

    def Predict(self, X_test):
        X_test = X_test - self.X_mean
        return np.matmul(X_test, self.parameters['beta'])
