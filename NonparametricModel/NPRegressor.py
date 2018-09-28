import numpy as np
import matplotlib.pyplot as plt
from ClassifierAndRegressor.ParametricModel import PRegressor as PR
from scipy.interpolate import UnivariateSpline


class UniSpline(PR.Regressor):
    def __init__(self, smooth_factor=None, poly_deg=3, **kwargs):
        super().__init__(**kwargs)
        self.smooth_factor = smooth_factor
        self.poly_deg = poly_deg
        self.x_sorted = None
        self.y_sorted = None

    def fit(self, x_train, y_train, sort=True):
        if len(y_train.shape) == 2:
            y_train = y_train.ravel()
        if len(x_train.shape) == 2:
            x_train = x_train.ravel()
        if sort:
            x_sorted, y_sorted = self.sort(x_train=x_train, y_train=y_train)
        else:
            x_sorted = x_train
            y_sorted = y_train

        self.x_sorted = x_sorted
        self.y_sorted = y_sorted
        self.regressor = UnivariateSpline(x=x_sorted, y=y_sorted, s=self.smooth_factor, k=self.poly_deg)

    def predict(self, x_test):
        try:
            return self.regressor(x_test)
        except TypeError:
            return self.regressor(x_test[0])

    def regression_plot(self, x_test=None, y_test=None):
        if x_test is None or y_test is None:
            x_test = self.x_sorted
            y_test = self.y_sorted
        else:
            x_test, y_test = self.sort(x_train=x_test, y_train=y_test)
        scatter = plt.scatter(x_test, y_test, color='b')
        line = plt.plot(x_test, self.predict(x_test), color='r')
        plt.legend(handles=[scatter, line[0], ], labels=['scatter plot', 'regression plot'], loc='best')
        plt.ylabel('response')
        plt.xlabel('explanatory')
        plt.title('Scatter Plot and Regression')
        plt.show()

    @classmethod
    def sort(cls, x_train, y_train):
        sorted_pair = zip(x_train, y_train)
        sorted_pair = sorted(sorted_pair)
        x_sorted = [x for x, _ in sorted_pair]
        y_sorted = [y for _, y in sorted_pair]
        while not all(np.diff(x_sorted) > 0):
            x_sorted = cls._check_diff(x_sorted)
            if len(np.where(np.diff(x_sorted) <= 0)[0]) == 0:
                break
        return x_sorted, y_sorted

    @classmethod
    def _check_diff(cls, x):
        diff = np.diff(x)
        if all(diff > 0):
            return x
        diff0 = np.where(diff == 0)[0]
        for index in list(diff0):
            x[index+1] = x[index] + 0.00000001
        return x
