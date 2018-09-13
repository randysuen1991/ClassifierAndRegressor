import numpy as np
import copy as cp

class CoreFun:
    @staticmethod
    def BackFitting(X_train, Y_train, smoother, threshold=0.1):
        alpha = np.mean(Y_train)
        smoothers = [smoother() for _ in range(X_train.shape[1])]
        while True:
            for i in range(X_train.shape[1]):
                y = None
                for j in range(X_train.shape[1]):
                    if j == i:
                        continue
                    y = Y_train - alpha - smoothers[j].predict(X_test=X_train[:, j])
                smoother[i].fit(X_train=X_train[:, i], Y_train=y)

            smoothers_old = cp.copy(smoothers)
            
    @staticmethod
    def _CheckConvergence(smoothers, smoothers_old, X_train, threshold):
        for smoother, smoother_old in zip(smoothers, smoothers_old):
            results_old = smoothers_old.predict(X_train)
            results = smoother.predict(X_train)
            sum = np.sum(results-results_old)
            if sum > threshold:
                return False
        return True