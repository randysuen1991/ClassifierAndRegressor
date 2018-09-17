import numpy as np


class Classifier:
    def __init__(self):
        self.parameters = None
        self.classifier = None
        self.x_k = None
        self.n = None
        self.y_k = None
        self._X_train = None
        self._Y_train = None
        self.recall = None
        self.precision = None
        self.accuracy = None
        self.valid_recall = None
        self.valid_precision = None
        self.valid_accuracy = None

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, X_train):
        self._X_train = X_train
        try:
            self.x_k = X_train.shape[1]
            self.n = X_train.shape[0]
        except IndexError:
            self.x_k = 1
            self.n = X_train.shape[0]
            self._X_train = self._X_train.reshape(-1, 1)

    @property
    def Y_train(self):
        return self._Y_train

    @Y_train.setter
    def Y_train(self, Y_train):
        self._Y_train = Y_train
        try:
            self.y_k = Y_train.shape[1]
        except IndexError:
            self.y_k = Y_train.shape[0]
            self._Y_train = self._Y_train.reshape(-1, 1)

    def Fit(self, X_train, Y_train, **kwargs):
        self.X_train = X_train
        self.Y_train = Y_train
        self._Inference(X_train, Y_train)
        self.classifier.fit(X_train, Y_train.ravel())

    def Classify(self, X_test, Y_test=None):
        try:
            results = self.classifier.predict(X_test)
        except AttributeError:
            return self.classifier.Classify(X_test, Y_test)

        if type(Y_test) != np.ndarray:
            return None, results, None
        else:
            correct_results = np.where(results == Y_test.ravel())[0]
            return len(correct_results) / len(Y_test), results, correct_results

    def Evaluate(self, Y_pred):
        if len(np.unique(self.Y_train)) == 2:
            positive = np.unique(self.Y_train)[0]
            negative = np.unique(self.Y_train)[1]
            Y_train = self.Y_train.ravel()
            valid_results = np.array(Y_pred)
            pred_positive = np.where(valid_results == positive)[0]
            pred_negative = np.where(valid_results == negative)[0]
            label_positive = np.where(Y_train == positive)[0]
            label_negative = np.where(Y_train == negative)[0]
            true_positive = np.intersect1d(pred_positive, label_positive)
            false_positive = np.intersect1d(pred_positive, label_negative)
            false_negative = np.intersect1d(pred_negative, label_positive)
            TP = len(true_positive)
            FP = len(false_positive)
            FN = len(false_negative)
            P = TP + FN
            recall = TP / P
            try:
                precision = TP / (TP + FP)
            except ZeroDivisionError:
                precision = 0
            return recall, precision
        else:
            return None, None

    # This function does 1/3-folded cross-validation.
    def _Inference(self, X_train, Y_train):
        n = X_train.shape[0]
        perm_X_train = np.random.permutation(X_train)
        valid_num = int(n / 3)
        X_valid = perm_X_train[0:valid_num, :]
        X_train = perm_X_train[valid_num:, :]
        Y_valid = Y_train[0:valid_num, :]
        Y_train = Y_train[valid_num:, :]
        try:
            self.classifier.Fit(X_train, Y_train)
        except AttributeError:
            self.classifier.fit(X_train, Y_train.ravel())
        self.valid_accuracy, valid_results, _ = self.Classify(X_valid, Y_valid.ravel())
        self.valid_recall, self.valid_precision = self.Evaluate(valid_results)
