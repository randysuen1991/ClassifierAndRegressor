import numpy as np
import copy as cp
from DataAnalysis import DataAnalysis as DA


class AdditiveModel:
    def __init__(self, smoother, smoother_factor):
        self.smoother = smoother
        self.smoother_factor = smoother_factor
        self.alpha = None
        self.smoothers = None

    def fit(self, x_train, y_train, threshold):
        self.alpha, self.smoothers = self.backfitting(x_train=x_train, y_train=y_train, smoother=self.smoother,
                                                      smooth_factor=self.smoother_factor, threshold=threshold)

    def predict(self, x_test, show_all=False):
        results = self.alpha
        results_list = list()
        for iterator, smoother in enumerate(self.smoothers):
            result = smoother.predict(x_test[:, iterator])
            results += result
            results_list.append(round(result[0], 2))
        if show_all:
            print('results:')
            print(results_list)
        return results

    @classmethod
    def backfitting(cls, x_train, y_train, smoother, smooth_factor, threshold=0.01, plot=False, outlierremoving=False):
        y_train = y_train.ravel()
        means = np.zeros(x_train.shape[1])
        alpha = np.mean(y_train)
        smoothers = [smoother(smooth_factor=smooth_factor) for _ in range(x_train.shape[1])]
        first = True
        smoothers_old = None
        c = 0
        # outliers removing
        if outlierremoving:
            index = DA.DataAnalysis.outlierremoving(x_train=x_train, num_of_std=3)
            index_unique = np.unique(index[0])
            x_train = np.delete(x_train, index_unique, axis=0)
            y_train = np.delete(y_train, index_unique, axis=0)
        while True:
            for i in range(x_train.shape[1]):
                y = y_train - alpha
                for j in range(x_train.shape[1]):
                    if j != i and (c > 0 or j < i):
                        y -= (smoothers[j].predict(x_test=x_train[:, j]) - means[j])
                smoothers[i].fit(x_train=x_train[:, i], y_train=y)
                means[i] = np.mean(smoothers[i].predict(x_train[:, i]))
            if not first:
                if cls._check_convergence(smoothers, smoothers_old, x_train, threshold):
                    # print('Convergence!')
                    if plot:
                        for i, smoother in enumerate(smoothers):
                            print('i:', i)
                            smoother.scatter_plot()
                            smoother.regression_plot()
                    return alpha, smoothers

            smoothers_old = cp.deepcopy(smoothers)
            first = False
            c += 1
            if c == 20000:
                # print('Not convergence, but number of iteration times exceeds.')
                return alpha, smoothers

    @staticmethod
    def _check_convergence(smoothers, smoothers_old, x_train, threshold):
        i = 0
        res = 0
        for smoother, smoother_old in zip(smoothers, smoothers_old):
            results_old = smoother_old.predict(x_train[:, i])
            results = smoother.predict(x_train[:, i])
            res += np.sum(np.abs(results-results_old))
            i += 1
        print('Residual:', np.abs(res))
        if np.abs(res) > threshold:
            return False
        return True
