import numpy as np
from scipy.interpolate import UnivariateSpline


class UniSpline(UnivariateSpline):
    def __init__(self, s=None, poly_deg=3, **kwargs):
        self.regressor = UnivariateSpline
        self.smooth_factor = s
        self.poly_deg = poly_deg

    def fit(self, X_train, Y_train):
        self.regressor(x=X_train, y=Y_train, s=self.smooth_factor, k=self.poly_deg)

