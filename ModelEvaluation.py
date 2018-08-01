import numpy as np

# The decorated function should either be passed by trained model with a estimated variance or a untrained model with
# training data. In the former case, it will return a list of values. (To avoid too many predictors, I use generator).
# In the latter case, it would return a value.


def PredictionErrorDecorator(fun):

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
                    yield fun(model, var)

            except TypeError:
                raise TypeError('Please specify the arguments: X_train, Y_train, k.')

        else:
            var = kwargs.get('var', None)
            if var is None:
                raise ValueError('You should give the estimated variance or give the not train model with the full '
                                 'set of predictors.')
            yield fun(model, var)

    return decofun


class ModelEvaluation:

    @PredictionErrorDecorator
    def AIC(model, var):
        return (model.sse + 2 * model.p * var) / model.n * var

    @PredictionErrorDecorator
    def BIC(model, var):
        return (model.sse + np.log(model.n) * model.p * var) / model.n

    @PredictionErrorDecorator
    def MallowCp(model, var):
        return (model.sse + 2 * model.x_k * var) / model.n

    def Rsquared(model):
        return 1 - model.sse / model.sst

    def AdjRsquared(model):
        return 1 - (1-ModelEvaluation.Rsquared(model)) * (model.n - 1) / (model.n - model.x_k - 1)
