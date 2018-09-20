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
        for iterator, smoother in enumerate(self.smoothers):
            # print(results)
            # print(smoother.predict(x_test[:, iterator]))
            results += smoother.predict(x_test[:, iterator])
        return results

    @classmethod
    def backfitting(cls, x_train, y_train, smoother, smooth_factor, threshold=0.1):
        alpha = np.mean(y_train)
        smoothers = [smoother(s=smooth_factor) for _ in range(x_train.shape[1])]
        first = True
        c = 0
        while True:
            for i in range(x_train.shape[1]):
                # print(i)
                # print('hhh')
                y = y_train - alpha
                for j in range(x_train.shape[1]):
                    # print(j)
                    if j == i:
                        continue
                    try:
                        y -= np.expand_dims(smoothers_old[j].predict(x_test=x_train[:, j]), axis=1)
                    except UnboundLocalError:
                        pass
                smoothers[i].fit(x_train=x_train[:, i], y_train=y, center=True)
            if not first:
                if cls._check_convergence(smoothers, smoothers_old, x_train, threshold):
                    print(c)
                    return alpha, smoothers

            smoothers_old = cp.deepcopy(smoothers)
            first = False
            c += 1

    @staticmethod
    def _check_convergence(smoothers, smoothers_old, x_train, threshold):
        for smoother, smoother_old in zip(smoothers, smoothers_old):
            results_old = smoother_old.predict(x_train)
            results = smoother.predict(x_train)
            res_sum = np.sum(np.abs(results-results_old))
            print(res_sum)
            if res_sum > threshold:
                return False
        return True
