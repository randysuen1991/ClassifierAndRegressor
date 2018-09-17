import numpy as np
from ClassifierAndRegressor.AdditiveModel import AdditiveModel as AM
from ClassifierAndRegressor.NonParametricModel import NPRegressor as NPR


def example1():
    model = AM.AdditiveModel(smoother=NPR.UniSpline, smoother_factor=0.5)


if __name__ == '__main__':
    example1()