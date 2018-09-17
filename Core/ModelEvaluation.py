import numpy as np
from functools import wraps
# The decorated function should either be passed by trained model with a estimated variance or a untrained model with
# training data. In the former case, it will return a list of values. (To avoid too many predictors, I use generator).
# In the latter case, it would return a value.


def PredictionErrorDecorator(fun):

    @wraps(fun)
    def decofun(model, **kwargs):
        if model.X_train is None:
            X_train = kwargs.get('X_train')
            Y_train = kwargs.get('Y_train')
            k = kwargs.get('k')
            model.Fit(X_train=X_train, Y_train=Y_train)
            var = model.sse / model.n
            try:
                for i in range(1, k+1):
                    predictors = X_train[:, 0:i]
                    # refit the model.
                    model.Fit(X_train=predictors, Y_train=Y_train)
                    return fun(model, var)

            except TypeError:
                raise TypeError('Please specify the arguments: X_train, Y_train, k.')

        else:
            var = kwargs.get('var', None)
            if var is None:
                raise ValueError('You should give the estimated variance or give the not train model with the full '
                                 'set of predictors.')
            return fun(model, var)

    return decofun


class ModelEvaluation:

    @staticmethod
    @PredictionErrorDecorator
    def AIC(model, var):
        yield (model.sse + 2 * model.x_k * var) / model.n * var

    @staticmethod
    @PredictionErrorDecorator
    def BIC(model, var):
        yield (model.sse + np.log(model.n) * model.x_k * var) / model.n

    @staticmethod
    @PredictionErrorDecorator
    def MallowCp(model, var):
        yield (model.sse + 2 * model.x_k * var) / model.n

    # FBeta is an indicator to evaluate the performance of a model.
    @staticmethod
    def ValidationFBeta(model, beta=0.5):
        try:
            return (1+beta**2) * model.valid_precision * model.valid_recall / \
                   (beta**2 * model.valid_precision + model.valid_recall)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def ValidationAccuracy(model):
        return model.valid_accuracy

    @staticmethod
    def Rsquared(model):
        return 1 - model.sse / model.sst

    @staticmethod
    def AdjRsquared(model):
        return 1 - (1-model.rsquared) * (model.n - 1) / (model.n - model.x_k - 1)
