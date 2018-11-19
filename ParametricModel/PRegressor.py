from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, LinearRegression, Lars
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from yellowbrick.regressor import ResidualsPlot
import yellowbrick

from sliced import SlicedInverseRegression
from ClassifierAndRegressor.Core import ModelEvaluation as ME
from ClassifierAndRegressor.Core import ModelSelection as MS
from ClassifierAndRegressor.Core.Regressor import Regressor
from DimensionReduction.DimensionReductionApproaches import centering_decorator, standardizing_decorator
from pyfinance.ols import PandasRollingOLS, RollingOLS


class OrdinaryLeastSquaredRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = LinearRegression()


class PartialLeastSquareRegressor(Regressor):
    def __init__(self, n_components):
        super().__init__()
        self.regressor = PLSRegression(n_components=n_components)

    def fit(self, x_train, y_train):
        self.regressor.fit(x_train, y_train)
        self.y_train = y_train
        self.x_train = x_train
        self._inference()
        return None, self.regressor.coef_, self.p, self.regressor.score(x_train, y_train)


class LassoRegressor(Regressor):
    def __init__(self, alpha):
        super().__init__()
        self.regressor = Lasso(alpha)

    def predict(self,  x_test):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        return np.expand_dims(self.regressor.predict(X=x_test), 1)


# Still need to check this class.
class PrincipalComponentRegressor(Regressor):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.regressor = LinearRegression()
        self.pca = None

    def fit(self, x_train, y_train, standardize=False):
        self.pca = PCA(self.n_components)
        self.x_train = self.pca.fit_transform(x_train)
        self.y_train = y_train
        self.regressor.fit(self.x_train, self.y_train)
        self._inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(self.x_train, y_train)

    def predict(self, x_test):
        try:
            x_test_transform = self.pca.transform(x_test)
        except ValueError:
            x_test_transform = x_test
        prediction = self.regressor.predict(x_test_transform)
        return prediction

    def residual_plot(self, x_test=None, y_test=None):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        try:
            self.residual_visualizer = ResidualsPlot(self.regressor)
        except yellowbrick.exceptions.YellowbrickTypeError:
            self.residual_visualizer = ResidualsPlot(self.regressor.regressor)

        self.residual_visualizer.fit(self.x_train, self.y_train)
        if x_test is not None and y_test is not None:
            try:
                self.residual_visualizer.score(x_test, y_test)
            except ValueError:
                x_test = self.pca.transform(x_test)
                self.residual_visualizer.score(x_test, y_test)
        self.residual_visualizer.poof()


class RidgeRegressor(Regressor):
    def __init__(self, alpha):
        super().__init__()
        self.regressor = Ridge(alpha)
        
        
class RandForestRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.regressor = RandomForestRegressor()

    def fit(self, x_train, y_train, standardize=False):
        self.standardize = standardize
        if self.standardize:
            self.standardizescaler.fit(x_train)
            x_train = self.standardizescaler.transform(x_train)

        self.x_train = x_train
        self.y_train = y_train
        self.regressor.fit(self.x_train, self.y_train.ravel())
        self._inference()
        return self.rsquared

    def residual_plot(self, x_test=None, y_test=None):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        try:
            self.residual_visualizer = ResidualsPlot(self.regressor)
        except yellowbrick.exceptions.YellowbrickTypeError:
            self.residual_visualizer = ResidualsPlot(self.regressor.regressor)

        y_train = self.y_train.ravel()
        self.residual_visualizer.fit(self.x_train, y_train)
        if x_test is not None and y_test is not None:
            y_test = y_test.ravel()
            self.residual_visualizer.score(x_test, y_test)
        self.residual_visualizer.poof()

    def predict(self, x_test):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        return self.regressor.predict(x_test).reshape(-1, 1)


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

    def fit(self, x_train, y_train, standardize=False, **kwargs):
        self.standardize = standardize

        self.pred_ind = MS.ModelSelection.ForwardSelection(model=self.regressor_type, x_train=x_train,
                                                           y_train=y_train, p=kwargs.get('p', x_train.shape[1]),
                                                           standardize=self.standardize)
        if self.standardize:
            self.standardizescaler.fit(x_train[:, self.pred_ind])
            self.x_train = self.standardizescaler.transform(x_train[:, self.pred_ind])
        else:
            self.x_train = x_train[:, self.pred_ind]

        self.y_train = y_train

        try:
            self.regressor.fit(self.x_train, self.y_train)
        except AttributeError:
            self.regressor.fit(self.x_train, self.y_train)

        self._inference()
        try:
            return self.regressor.intercept_, self.regressor.coef_, self.p, \
                self.regressor.score(self.x_train, self.y_train), self.pred_ind
        except AttributeError:
            return self.regressor.regressor.intercept_, self.regressor.regressor.coef_, self.p, \
                   self.regressor.regressor.score(self.x_train, self.y_train), self.pred_ind


class BackwardStepwiseRegressor(Regressor):
    def __init__(self, regressor=OrdinaryLeastSquaredRegressor, criteria=ME.ModelEvaluation.AIC):
        super().__init__()
        self.regressor_type = regressor
        self.regressor = regressor()
        self.criteria = criteria
        self.pred_ind = None

    def fit(self, x_train, y_train, standardize=False, **kwargs):
        self.standardize = standardize
        self.pred_ind = MS.ModelSelection.BackwardSelection(model=self.regressor_type, x_train=x_train,
                                                            y_train=y_train, p=kwargs.get('p', x_train.shape[1]),
                                                            standardize=self.standardize)
        if self.standardize:
            self.x_train = self.standardizescaler.fit_transform(x_train[:, self.pred_ind])
        else:
            self.x_train = x_train[:, self.pred_ind]

        self.y_train = y_train
        self.regressor.fit(self.x_train, self.y_train)
        self._inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, \
            self.regressor.score(self.x_train, self.y_train), self.pred_ind


class BestsubsetRegressor(Regressor):
    def __init__(self, criteria=ME.ModelEvaluation.AIC):
        super().__init__()
        self.regressor = LinearRegression()
        self.criteria = criteria

    def fit(self, x_train, y_train, **kwargs):
        ids = MS.ModelSelection.BestSubsetSelection(model=OrdinaryLeastSquaredRegressor, x_train=x_train,
                                                    y_train=y_train, p=kwargs.get('p', x_train.shape[1]))
        self.x_train = x_train[:, ids]
        self.y_train = y_train
        self.regressor.fit(self.x_train, self.y_train)
        self._inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, \
               self.regressor.score(self.x_train, self.y_train), ids


# The response should be univariate.
class ForwardStagewiseRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self.X_mean = None
        self.Y_mean = None

    @centering_decorator('x_train', 'y_train')
    @standardizing_decorator('x_train', 'y_train')
    def fit(self, x_train, y_train, **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.X_mean = np.mean(x_train, axis=0)
        self.Y_mean = np.mean(y_train, axis=0)
        eps = kwargs.get('eps', 0.01)
        k = kwargs.get('k', self.x_k)
        lower_bound = kwargs.get('lower_bound', 0)
        assert k <= self.x_k
        residual = y_train
        available_predictors = list(range(k))
        cors = np.zeros(shape=(k, ))
        beta = np.zeros(shape=(k, ))
        for i in range(k):
            for predictor in available_predictors:
                cors[predictor] = np.matmul(residual.T, x_train[:, predictor])
            abs_cors = [np.abs(cor) for cor in cors]
            index = np.argmax(abs_cors)
            if abs_cors[index] < lower_bound:
                break
            sign = np.sign(cors[index])
            beta[index] += sign * eps
            residual -= sign * eps * np.expand_dims(x_train[:, index], axis=1)

            available_predictors.remove(index)
            cors[index] = 0
            abs_cors[index] = 0

        self.parameters['beta'] = beta
        self._inference(x_train, y_train)

        return 0, self.parameters['beta'], self.p, self.rsquared

    def predict(self, x_test):
        x_test = x_test - self.X_mean
        return np.matmul(x_test, self.parameters['beta'])


# I add "predict" method to the RollingOLS class.
class ExtendedRollingOLS(Regressor):
    def __init__(self, window_size=None, has_const=False, use_const=True):
        super().__init__()
        self.window_size = window_size
        self.has_const = has_const
        self.use_const = use_const

    def fit(self, x_train, y_train, standardize=False):
        self.x_train = x_train
        self.y_train = y_train
        self.standardize = standardize
        if self.standardize:
            self.standardizescaler.fit(x_train)
            x_train = self.standardizescaler.transform(x_train)

        self.regressor = RollingOLS(y=y_train, x=x_train, window=self.window_size, has_const=self.has_const,
                                    use_const=self.use_const)

    def predict(self, x_test):
        parameters = np.hstack((self.regressor.alpha.values.reshape(-1,1), self.regressor.beta.values))
        extended_x_test = np.hstack((np.ones(shape=(x_test.shape[0], 1)), x_test))
        return np.matmul(extended_x_test, parameters)


class ExtendedPandasRollingOLS(ExtendedRollingOLS):
    def __init__(self, window_size=None, has_const=False, use_const=True):
        super().__init__(window_size, has_const, use_const)

    def fit(self, x_train, y_train, standardize=False):
        self.x_train = x_train
        self.y_train = y_train
        self.standardize = standardize
        if self.standardize:
            self.standardizescaler.fit(x_train)
            x_train = self.standardizescaler.transform(x_train)

        self.regressor = PandasRollingOLS(y=y_train, x=x_train, window=self.window_size, has_const=self.has_const,
                                          use_const=self.use_const)