from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import ResidualsPlot
import yellowbrick
from ClassifierAndRegressor.Core import ModelEvaluation as ME


class Regressor:
    def __init__(self):
        self.parameters = dict()
        self.regressor = None
        self.sse = None
        self.sst = None
        self.adjrsquared = None
        self.rsquared = None
        self._x_train = None
        self._y_train = None
        self.n = None
        # x_k refers to the number of the predictors of x_train
        self.x_k = None
        # y_k refers to the number of the responses of y_train
        self.y_k = None
        self.p = None
        self.standardize = None
        self.standardizescaler = StandardScaler()
        self.residual_visualizer = None

    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, x_train):
        self._x_train = x_train
        try:
            self.x_k = x_train.shape[1]
            self.n = x_train.shape[0]
        except IndexError:
            self.x_k = 1
            self.n = x_train.shape[0]
            self._x_train = self._x_train.reshape(-1, 1)

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, y_train):
        self._y_train = y_train
        try:
            self.y_k = y_train.shape[1]
        except IndexError:
            self.y_k = y_train.shape[0]
            self._y_train = self._y_train.reshape(-1, 1)

    def _inference(self):

        try:
            self.rsquared = self.regressor.score(self.x_train, self.y_train)
        except AttributeError:
            self.rsquared = self.regressor.regressor.score(self.x_train, self.y_train)

        self.adjrsquared = ME.ModelEvaluation.AdjRsquared(self)

        # Store some info of the model.
        self.sst = np.sum((self.y_train-np.mean(self.y_train, axis=0))**2, axis=0)
        self.sse = np.sum((self.predict(self.x_train) - self.y_train) ** 2, axis=0)
        self.sse_scaled = self.sse / float(self.x_train.shape[0] - self.x_train.shape[1])

        if type(self.sse_scaled) == np.float64:
            self.sse_scaled = [self.sse_scaled]
        try:
            if not self.standardize:
                x_train = self.x_train - np.mean(self.x_train, axis=0)
            else:
                x_train = self.x_train
            var_beta = self.sse_scaled * (np.linalg.inv(np.dot(x_train.T, x_train)).diagonal())
            self.se = np.sqrt(var_beta)
        except np.linalg.linalg.LinAlgError:
            return
        except TypeError:
            return

        try:
            self.t = self.regressor.coef_ / np.array(self.se)
        except AttributeError:
            try:
                self.t = self.parameters['beta'] / self.se
            except KeyError:
                return
        self.p = [2 * (1-stats.t.cdf(np.abs(i), (len(x_train)-1))) for i in self.t]

    def fit(self, x_train, y_train, standardize=False):
        x_train = x_train
        y_train = y_train

        self.standardize = standardize
        if self.standardize:
            self.standardizescaler.fit(x_train)
            x_train = self.standardizescaler.transform(x_train)
        self.x_train = x_train
        self.y_train = y_train
        self.regressor.fit(self.x_train, self.y_train)
        self._inference()
        return self.regressor.intercept_, self.regressor.coef_, self.p, self.regressor.score(x_train, y_train)

    def predict(self, x_test):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        try:
            return self.regressor.predict(x_test)
        except AttributeError:
            return self.regressor.predict(x_test=x_test)

    def regression_plot(self, x_test, y_test):
        scatter = plt.scatter(x_test, y_test, color='b')
        try:
            line = plt.plot(x_test, self.regressor.predict(x_test), color='r')
        except AttributeError:
            line = plt.plot(x_test, self.regressor.predict(x_test), color='r')
        plt.ylabel('response')
        plt.xlabel('explanatory')
        plt.legend(handles=[scatter, line[0], ],
                   labels=['Scatter Plot',
                           'Intercept:{}, Slope:{},\n R-square:{}'.format(self.regressor.intercept_,
                                                                          self.regressor.coef_,
                                                                          self.regressor.score(x_test, y_test))],
                   loc='best')
        plt.title('Scatter Plot and Regression')

    def residual_plot(self, x_test=None, y_test=None):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        try:
            self.residual_visualizer = ResidualsPlot(self.regressor)
        except yellowbrick.exceptions.YellowbrickTypeError:
            self.residual_visualizer = ResidualsPlot(self.regressor.regressor)

        self.residual_visualizer.fit(self.x_train, self.y_train)
        if x_test is not None and y_test is not None:
            self.residual_visualizer.score(x_test, y_test)
        self.residual_visualizer.poof()

    def get_score(self, x_test, y_test):
        if self.standardize:
            x_test = self.standardizescaler.transform(x_test)
        try:
            return self.regressor.score(x_test, y_test)
        except AttributeError:
            return self.regressor.Get_Score(x_test, y_test)

