import numpy as np
from ClassifierAndRegressor.ParametricModel import PRegressor as PR
from scipy.interpolate import UnivariateSpline


class UniSpline(PR.Regressor):
    def __init__(self, s=None, poly_deg=3, **kwargs):
        super().__init__(**kwargs)
        self.smooth_factor = s
        self.poly_deg = poly_deg
        self.centered = False
        self.x_train_mean = None

    def fit(self, x_train, y_train, center=False, sort=True):
        if len(y_train.shape) == 2:
            y_train = y_train.ravel()
        if len(x_train.shape) == 2:
            x_train = x_train.ravel()
        if sort:
            sorted_pair = zip(x_train, y_train)
            sorted_pair = sorted(sorted_pair)
            x_sorted = [x for x, _ in sorted_pair]
            y_sorted = [y for _, y in sorted_pair]
            while not all(np.diff(x_sorted) > 0):
                x_sorted = self._check_diff(x_sorted)
                if len(np.where(np.diff(x_sorted) <= 0)[0]) == 0:
                    break
        else:
            x_sorted = x_train
            y_sorted = y_train
        # print('real fit!')
        self.regressor = UnivariateSpline(x=x_sorted, y=y_sorted, s=self.smooth_factor, k=self.poly_deg)
        # print('real fit over!')
        if center:
            self.x_train_mean = np.mean(self.regressor(x_sorted))
            print('mean:', self.x_train_mean)
            self.centered = True
        else:
            self.centered = False

    def predict(self, x_test):
        if self.centered:
            results = self.regressor(x_test) - self.x_train_mean
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
