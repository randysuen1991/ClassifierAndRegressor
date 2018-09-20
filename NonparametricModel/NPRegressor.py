import numpy as np
from ClassifierAndRegressor.ParametricModel import PRegressor as PR
from scipy.interpolate import UnivariateSpline


class UniSpline(PR.Regressor):
    def __init__(self, s=None, poly_deg=3, **kwargs):
        super().__init__(**kwargs)
        self.smooth_factor = s
        self.poly_deg = poly_deg
        self.centered = False
        self.X_train_mean = None

    def fit(self, x_train, y_train, center=False):
        if len(y_train.shape) == 2:
            y_train = y_train.ravel()
        if len(x_train.shape) == 2:
            x_train = x_train.ravel()
        sorted_pair = zip(x_train, y_train)
        sorted_pair = sorted(sorted_pair)
        x_sorted = [x for x, _ in sorted_pair]
        y_sorted = [y for _, y in sorted_pair]
        while not all(np.diff(x_sorted) > 0):
            x_sorted = self._check_diff(x_sorted)
            if len(np.where(np.diff(x_sorted) <= 0)[0]) == 0:
                break
        self.regressor = UnivariateSpline(x=x_sorted, y=y_sorted, s=self.smooth_factor, k=self.poly_deg)
        if center:
            self.X_train_mean = np.mean(self.regressor(x_train))
            self.centered = True
        else:
            self.centered = False

    def predict(self, x_test):
        if self.centered:
            # print('prediction:', self.regressor(x_test).shape)
            # print('mean:', self.X_train_mean)
            results = self.regressor(x_test) - self.X_train_mean
        else:
            results = self.regressor(x_test)
        return results

    @classmethod
    def _check_diff(cls, x):
        diff = np.diff(x)
        if all(diff > 0):
            return x
        diff0 = np.where(diff == 0)[0]
        for index in list(diff0):
            x[index+1] = x[index] + 0.00000001
        return x
