import numpy as np

class CoreFun:

    @staticmethod
    def BackFitting(X_train, Y_train, smoother):
        alpha = np.mean(Y_train)
        while True:
            