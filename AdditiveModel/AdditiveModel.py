import numpy as np
import copy as cp


class AdditiveModel:
    def __init__(self, smoother, smoother_factor):
        self.smoother = smoother
        self.smoother_factor = smoother_factor
        self.alpha = None
        self.smoothers = None

    def fit(self, x_train, y_train):
        self.alpha, self.smoothers = self.backfitting(x_train=x_train, y_train=y_train, smoother=self.smoother,
                                                      smooth_factor=self.smoother_factor)

    def predict(self, x_test):
        results = self.alpha
        for smoother in self.smoothers:
            results += smoother(x_test)
        return results

    @classmethod
    def backfitting(cls, x_train, y_train, smoother, smooth_factor, threshold=0.1):
        alpha = np.mean(y_train)
        smoothers = [smoother(smooth_factor=smooth_factor) for _ in range(x_train.shape[1])]
        first = True
        while True:
            for i in range(x_train.shape[1]):
                y = y_train - alpha
                for j in range(x_train.shape[1]):
                    if j == i:
                        continue
                    try:
                        y -= smoothers[j].predict(X_test=x_train[:, j])
                    except TypeError:
                        pass
                smoother[i].fit(X_train=x_train[:, i], Y_train=y, center=True)

            if not first:
                if cls._check_convergence(smoothers, smoothers_old, x_train, threshold):
                    return alpha, smoothers

            smoothers_old = cp.deepcopy(smoothers)
            first = False

    @staticmethod
    def _check_convergence(smoothers, smoothers_old, x_train, threshold):
        for smoother, smoother_old in zip(smoothers, smoothers_old):
            results_old = smoothers_old.predict(x_train)
            results = smoother.predict(x_train)
            sum = np.sum(results-results_old)
            if sum > threshold:
                return False
        return True
