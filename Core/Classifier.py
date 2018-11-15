import numpy as np


class Classifier:
    def __init__(self):
        self.parameters = None
        self.classifier = None
        self.x_k = None
        self.n = None
        self.y_k = None
        self._x_train = None
        self._y_train = None
        self.recall = None
        self.precision = None
        self.accuracy = None
        self.valid_recall = None
        self.valid_precision = None
        self.valid_accuracy = None

    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, x_train):
        self._x_train = x_train
        try:
            self.x_k = x_train.shape[1]
            self.n = x_train.shape[0]
        except IndexError:
            self.x_k = 1
            self.n = x_train.shape[0]
            self._x_train = self._x_train.reshape(-1, 1)

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, y_train):
        self._y_train = y_train
        try:
            self.y_k = y_train.shape[1]
        except IndexError:
            self.y_k = y_train.shape[0]
            self._y_train = self._y_train.reshape(-1, 1)

    def fit(self, x_train, y_train, **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self._inference(x_train, y_train)
        self.classifier.fit(x_train, y_train.ravel())

    def classify(self, x_test, y_test=None):
        try:
            results = self.classifier.predict(x_test)
        except AttributeError:
            return self.classifier.classify(x_test, y_test)

        if type(y_test) != np.ndarray:
            return None, results, None
        else:
            correct_results = np.where(results == y_test.ravel())[0]
            return len(correct_results) / len(y_test), results, correct_results

    def Evaluate(self, Y_pred):
        if len(np.unique(self.y_train)) == 2:
            positive = np.unique(self.y_train)[0]
            negative = np.unique(self.y_train)[1]
            y_train = self.y_train.ravel()
            valid_results = np.array(Y_pred)
            pred_positive = np.where(valid_results == positive)[0]
            pred_negative = np.where(valid_results == negative)[0]
            label_positive = np.where(y_train == positive)[0]
            label_negative = np.where(y_train == negative)[0]
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
    def _inference(self, x_train, y_train):
        n = x_train.shape[0]
        perm_x_train = np.random.permutation(x_train)
        valid_num = int(n / 3)
        X_valid = perm_x_train[0:valid_num, :]
        x_train = perm_x_train[valid_num:, :]
        Y_valid = y_train[0:valid_num, :]
        y_train = y_train[valid_num:, :]
        try:
            self.classifier.fit(x_train, y_train)
        except AttributeError:
            self.classifier.fit(x_train, y_train.ravel())
        self.valid_accuracy, valid_results, _ = self.classify(X_valid, Y_valid.ravel())
        self.valid_recall, self.valid_precision = self.Evaluate(valid_results)
