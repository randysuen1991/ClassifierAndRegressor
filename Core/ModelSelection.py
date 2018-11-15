import numpy as np
import warnings
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from ClassifierAndRegressor.Core import ModelEvaluation as ME
from ClassifierAndRegressor.ParametricModel import PRegressor as PR
from ClassifierAndRegressor.ParametricModel import PClassifier as PC
from ClassifierAndRegressor.Core import Regressor as R
from ClassifierAndRegressor.Core import Classifier as C
from DimensionReduction.DimensionReductionApproaches import CenteringDecorator, StandardizingDecorator


class VariableSelection:
    # Till now, y_train should be a N*1 matrix.
    @staticmethod
    @CenteringDecorator('x_train', 'y_train')
    def corrselection(x_train, y_train, both_sides, num_each_side, abs=False, **kwargs):
        if y_train.shape[1] > 1:
            warnings.warn('The dimension of the Y variable should be 1 now.')
        covariance = np.matmul(x_train.T, y_train)/x_train.shape[0]
        X_std = np.expand_dims(np.std(x_train, axis=0), axis=1)
        Y_std = np.std(y_train)
        corr = covariance/(X_std*Y_std)
        
        if abs is True:
            if num_each_side == 'full':
                num_each_side = x_train.shape[1]
            corr = np.abs(corr)
            sorted_index_all = np.argsort(corr.ravel())
            sorted_index_all = sorted_index_all[::-1]
            sorted_left = sorted_index_all[0:num_each_side]
            sorted_index = sorted_left
            if both_sides is True:
                warnings.warn('You should not have picked the both sides of the variable list.')
            return sorted_index, np.reshape(corr[sorted_index], newshape=(1, len(corr[sorted_index])))
        
        sorted_index_all = np.argsort(corr.ravel())
        sorted_index_all = sorted_index_all[::-1]
        
        if both_sides is True:
            # if fill, it will discard the weakest one
            if num_each_side == 'full':
                num_each_side = np.floor(x_train.shape[1]/2)
                
            index_left = sorted_index_all[0:num_each_side]
            index_right = sorted_index_all[-1:-1-num_each_side:-1]
            sorted_index = np.concatenate([index_left, index_right], axis=0)
        else:
            if num_each_side == 'full':
                num_each_side = x_train.shape[1]
                
            index_left = sorted_index_all[0:num_each_side]
            sorted_index = index_left
            
        return sorted_index, np.reshape(corr[sorted_index], newshape=(1, len(corr[sorted_index])))


class ModelSelection:

    @staticmethod
    def bestsubsetselection(model, x_train, y_train, criteria=ME.ModelEvaluation.AIC, **kwargs):
        warnings.warn('Please notice that when the number of predictors are too large, the'
                      'best subset selection would be quite time-consuming.')
        p = kwargs.get('p', x_train.shape[1])
        candidates = list()
        predictors_candidates = list()
        for i in range(1, p):
            model_candidates = list()
            combs = list(combinations(range(p), i))
            for comb in combs:
                model_comb = model()
                model_comb.Fit(x_train=x_train[:, comb], y_train=y_train)
                model_candidates.append((model_comb, comb))

            if isinstance(model_candidates[0][0], R.Regressor):
                rsquareds = [_model.rsquared for _model, _ in model_candidates]
                index = np.argmax(rsquareds)
            elif isinstance(model_candidates[0][0], C.Classifier):
                numbers = [criteria(_model) for _model, _ in model_candidates]
                index = np.argmax(numbers)

            model_selected, predictor_id = model_candidates[index]

            candidates.append(model_selected)
            predictors_candidates.append(predictor_id)

        if criteria is ME.ModelEvaluation.Rsquared or criteria is ME.ModelEvaluation.AdjRsquared:
            numbers = [criteria(_model) for _model in candidates]
            index = np.argmax(numbers)
        elif criteria is ME.ModelEvaluation.AIC or criteria is ME.ModelEvaluation.BIC or \
                criteria is ME.ModelEvaluation.MallowCp:
            model_full = model()
            model_full.Fit(x_train=x_train, y_train=y_train)
            var = model_full.sse / model_full.n
            numbers = [list(criteria(_model, var=var))[0] for _model in candidates]
            index = np.argmin(numbers)
        elif criteria is ME.ModelEvaluation.ValidationAccuracy or criteria is ME.ModelEvaluation.ValidationFBeta:
            numbers = [criteria(_model) for _model in candidates]
            index = np.argmax(numbers)
        else:
            raise TypeError('Please handle the special criteria.')
        return predictors_candidates[index]

    # This function would return a list of indices indicating which predictors we should select.
    @staticmethod
    def forwardselection(model, x_train, y_train, criteria=ME.ModelEvaluation.MallowCp, **kwargs):
        p = kwargs.get('p', x_train.shape[1])
        if p > x_train.shape[0]:
            p = x_train.shape[0]
        if kwargs.get('standardize', False):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
        candidates = list()
        predictors_order = list()
        available_predictors = list(range(p))
        for i in range(p):
            model_candidates = list()
            for j in available_predictors:
                add_model = model()
                try:
                    tmp = x_train[:, j].reshape(-1, 1)
                    add_predictors = np.concatenate((predictors, tmp), axis=1)
                except UnboundLocalError:
                    add_predictors = x_train[:, j].reshape(-1, 1)

                try:
                    add_model.Fit(x_train=add_predictors, y_train=y_train)
                except AttributeError:
                    add_model.fit(add_predictors, y_train)

                model_candidates.append((add_model, j))

            if isinstance(model_candidates[0][0], R.Regressor):
                rsquareds = [_model.rsquared for _model, _ in model_candidates]
                index = np.argmax(rsquareds)
            elif isinstance(model_candidates[0][0], C.Classifier):
                numbers = [criteria(_model) for _model, _ in model_candidates]
                index = np.argmax(numbers)

            model_selected, predictor_id = model_candidates[index]
            available_predictors.remove(predictor_id)
            candidates.append(model_selected)
            predictors_order.append(predictor_id)
            predictors = model_selected.x_train

        if criteria is ME.ModelEvaluation.Rsquared or criteria is ME.ModelEvaluation.AdjRsquared:
            numbers = [criteria(_model) for _model in candidates]
            index = np.argmax(numbers)

        elif criteria is ME.ModelEvaluation.AIC or criteria is ME.ModelEvaluation.BIC or \
                criteria is ME.ModelEvaluation.MallowCp:
            model_full = model()
            model_full.Fit(x_train=x_train, y_train=y_train, standardize=kwargs.get('standardize', False))
            var = model_full.sse / model_full.n
            var = var[0]
            numbers = [list(criteria(_model, var=var))[0] for _model in candidates]
            index = np.argmin(numbers)

        elif criteria is ME.ModelEvaluation.ValidationAccuracy or criteria is ME.ModelEvaluation.ValidationFBeta:
            numbers = [criteria(_model) for _model in candidates]
            index = np.argmax(numbers)
        else:
            raise TypeError('Please handle the special criteria.')

        return predictors_order[:index+1]

    @staticmethod
    def backwardselection(model, x_train, y_train, criteria=ME.ModelEvaluation.AIC, **kwargs):
        p = kwargs.get('p', x_train.shape[1])
        x_train = x_train[:, :p]
        if x_train.shape[1] > x_train.shape[0]:
            tmp_model = model()
            if isinstance(tmp_model, R.Regressor):
                print(x_train.shape)
                raise ValueError('The number of predictors should not be larger the one of the sample size.')
            del tmp_model

        candidates = list()
        predictors_order = list()
        # this list stores the index of the predictors which are ok to be deleted.
        available_predictors = list(range(p))
        model_full = model()
        model_full.Fit(x_train=x_train, y_train=y_train, standardize=kwargs.get('standardize', False))
        candidates.append(model_full)
        for i in range(p-1):
            model_candidates = list()
            predictors = candidates[-1].x_train
            for j, k in enumerate(available_predictors):
                sub_model = model()
                sub_predictors = np.delete(predictors, j, axis=1)
                sub_model.Fit(x_train=sub_predictors, y_train=y_train)
                model_candidates.append((sub_model, k))

            if isinstance(model_candidates[0][0], R.Regressor):
                rsquareds = [_model.rsquared for _model, _ in model_candidates]
                index = np.argmax(rsquareds)
            elif isinstance(model_candidates[0][0], C.Classifier):
                numbers = [criteria(_model) for _model, _ in model_candidates]
                index = np.argmax(numbers)

            model_selected, predictor_id = model_candidates[index]

            available_predictors.remove(predictor_id)

            candidates.append(model_selected)
            predictors_order.append(predictor_id)
        if criteria is ME.ModelEvaluation.Rsquared or criteria is ME.ModelEvaluation.AdjRsquared:
            numbers = [criteria(_model) for _model in candidates]
            index = np.argmax(numbers)
        elif criteria is ME.ModelEvaluation.AIC or criteria is ME.ModelEvaluation.BIC or \
                criteria is ME.ModelEvaluation.MallowCp:
            model_full = model()
            model_full.Fit(x_train=x_train, y_train=y_train, standardize=kwargs.get('standardize', False))
            var = model_full.sse / model_full.n
            numbers = [list(criteria(_model, var=var))[0] for _model in candidates]
            index = np.argmin(numbers)
        elif criteria is ME.ModelEvaluation.ValidationAccuracy or criteria is ME.ModelEvaluation.ValidationFBeta:
            numbers = [criteria(_model) for _model in candidates]
            index = np.argmax(numbers)
        else:
            raise TypeError('Please handle the special criteria.')
        predictors = list(range(p))
        to_remove = predictors_order[:index]
        for predictor in to_remove:
            predictors.remove(predictor)
        return predictors
