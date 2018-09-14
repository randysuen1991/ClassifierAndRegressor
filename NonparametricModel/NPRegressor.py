import numpy as np
import sys
if 'C:\\Users\\randysuen\\pycodes\\Classifier-and-Regressor' not in sys.path:
    sys.path.append('C:\\Users\\randysuen\\pycodes\\Classifier-and-Regressor')
import Regressor as R
from scipy.interpolate import UnivariateSpline


class UniSpline(R.Regressor):
    def __init__(self, s=None, poly_deg=3, **kwargs):
        super().__init__(**kwargs)
        self.smooth_factor = s
        self.poly_deg = poly_deg
        self.centered = False
        self.X_train_mean = None

    def fit(self, x_train, y_train, center=False):
        self.X_train = x_train
        self.Y_train = y_train
        self.regressor = UnivariateSpline(x=self.X_train, y=self.Y_train, s=self.smooth_factor, k=self.poly_deg)
        if center:
            self.X_train_mean = np.mean(self.regressor(x_train))
            self.centered = True
        else:
            self.centered = False

    def predict(self, x_test):
        if self.centered:
            results = self.regressor(x_test) - self.X_train_mean
        else:
            results = self.regressor(x_test)
        return results
