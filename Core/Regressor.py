from scipy import stats
import numpy as np
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

