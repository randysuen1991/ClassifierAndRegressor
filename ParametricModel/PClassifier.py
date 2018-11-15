from DimensionReductionApproaches import UtilFun as UF
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression
import DimensionReductionApproaches as DRA
from ClassifierAndRegressor.Core import ModelEvaluation as ME
from ClassifierAndRegressor.Core import ModelSelection as MS
from ClassifierAndRegressor.Core.Classifier import Classifier

# I should have written a linear discriminant decoarator to decorate the method of LinearDiscriminantClassifier.
# i.e it oculd project the input argument into the discriminitive subspace.

# I also should have written a decorator for 'self.Classsify'. Since sometimes we don't have the labels of testing data.
# This is a failure decorator. I want to use it to decorate a class function, but it will miss 'self' argument.
"""
Notice :
The labels of the data, i.e the Y should always be numeric number larger or equal to 1
"""


class AdaBoostClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = Ada()
        self.kwargs = kwargs


class RandomForestClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = RF()
        self.kwargs = kwargs


class GaussianBayesClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = GaussianNB()
        self.kwargs = kwargs


class MultinomialBayesClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = MultinomialNB()


class BernoulliBayesClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = BernoulliNB()
        self.kwargs = kwargs


class GaussianProcessClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = GPC()
        self.kwargs = kwargs


class DecisionTreeClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = DTC()
        self.kwargs = kwargs


class LogisticClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = LogisticRegression()
        self.kwargs = kwargs


class KNearestNeighborClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = KNeighborsClassifier(n_neighbors=kwargs.get('k', 1))
        self.kwargs = kwargs


class SupportVectorClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = SVC()
        self.kwargs = kwargs
        

class LinearDiscriminantClassifier(Classifier):
    
    def __init__(self, discriminant_fun, classifier, **kwargs):
        super().__init__()
        self.classifier = classifier
        self.discriminant_function = discriminant_fun
        self.kwargs = kwargs

    def fit(self, x_train, y_train):
        self.parameters = self.discriminant_function(x_train=x_train, y_train=y_train, kwargs=self.kwargs)
        x_train_proj = np.matmul(x_train, self.parameters)
        self.classifier.fit(x_train_proj, y_train.ravel())
        return self.parameters

    def classify(self, x_test, y_test=None):
        x_test_proj = np.matmul(x_test, self.parameters)
        return self.classifier.classify(x_test_proj, y_test)


class ForwardStepwiseClassifier(Classifier):
    def __init__(self, classifier, **kwargs):
        super().__init__()
        self.classifier = classifier()
        self.classifier_type = classifier
        self.kwargs = kwargs

    def fit(self, x_train, y_train):

        ids = MS.ModelSelection.ForwardSelection(self.classifier_type, x_train, y_train,
                                                 criteria=ME.ModelEvaluation.ValidationFBeta)
        self.x_train = x_train[:, ids]
        self.y_train = y_train
        self._inference(x_train[:, ids], y_train)
        try:
            self.classifier.fit(self.x_train, self.y_train.ravel())
        except AttributeError:
            self.classifier.fit(self.x_train, self.y_train)

        return ids


class BackwardStepwiseClassifier(Classifier):
    def __init__(self, classifier, **kwargs):
        super().__init__()
        self.classifier = classifier()
        self.classifier_type = classifier
        self.kwargs = kwargs

    def fit(self, x_train, y_train):

        ids = MS.ModelSelection.BackwardSelection(self.classifier_type, x_train, y_train,
                                                  criteria=ME.ModelEvaluation.ValidationAccuracy)
        self.x_train = x_train[:, ids]
        self.y_train = y_train
        try:
            self.classifier.fit(self.x_train, self.y_train)
        except AttributeError:
            self.classifier.fit(self.x_train, self.y_train)

        return ids


class BestsubsetClassifier(Classifier):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier()
        self.classifier_type = classifier

    def fit(self, x_train, y_train):
        ids = MS.ModelSelection.BackwardSelection(self.classifier_type, x_train, y_train,
                                                  criteria=ME.ModelEvaluation.ValidationAccuracy)

        self.x_train = x_train[:, ids]
        self.y_train = y_train
        self.classifier.fit(self.x_train, self.y_train)


class TwoStepClassifier(Classifier):
    def __init__(self, first_step_function, second_step_function, classify_function, **kwargs):
        super().__init__()
        self.first_step_function = first_step_function
        self.second_step_function = second_step_function
        self.classify_function = classify_function
        self.parameters = dict()
        self.input_shape = None

    def fit(self, x_train, y_train, **kwargs):

        self.input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])
        dimension = self._Search_Dimensionality(x_train=x_train,
                                                y_train=y_train)
        linear_subspace = self.first_step_function(x_train=x_train,
                                                   y_train=y_train,
                                                   p_tilde=dimension,
                                                   q_tilde=dimension)
        
        self.parameters['first_step']['row'] = linear_subspace[0]
        self.parameters['first_step']['column'] = linear_subspace[1]
        
        x_train_proj = DRA.MultilinearReduction.TensorProject(x_train, self.parameters['first_step']['row'],
                                                              self.parameters['first_step']['column'])
        x_train_proj_vec = UF.imgs2vectors(x_train_proj)
        
        self.parameters['second_step'] = self.second_step_function(x_train=x_train,y_train=y_train)
        
        self.transformed_x_train = np.matmul(x_train_proj_vec,
                                             self.parameters['second_step'])
        self.y_train = y_train
        return self.parameters
    
    def _Search_Dimensionality(self, x_train, y_train, p_tilde, q_tilde, dimension=50):

        ratios = []
        for iter in range(dimension):
            linear_subspace = self.first_step_function(x_train=x_train, input_shape=self.input_shape,
                                                       p_tilde=iter+1, q_tilde=iter+1)
            x_train_proj = np.matmul(np.matmul(linear_subspace[0], x_train), linear_subspace[1])
            x_train_proj_vec = UF.imgs2vectors(x_train_proj)
            linear_subspace = self.second_step_function(x_train_proj, y_train)
            x_train_proj_vec_proj = np.matmul(x_train_proj_vec, linear_subspace)
            ratio = self.Compute_Ratio(x_train_proj_vec_proj, y_train)
            ratios.append(ratio)
        
        ratios = np.round(ratios,6)
        index = np.argmax(ratios)
        
        return index + 1

    def classify(self, x_test, y_test, **kwargs):
        k = kwargs.get('k', 1)
        # Use K-nearest neighbor to classify the testing data
        neighbor = KNeighborsClassifier(n_neighbors=k)
        neighbor.fit(self.transformed_x_train, self.y_train.ravel())
        
        x_test_proj = DRA.MultilinearReduction.TensorProject(x_test, self.parameters['first_step']['row'],
                                                             self.parameters['first_step']['column'])
        x_test_proj_vec = UF.imgs2vectors(x_test_proj)
        x_test_proj_vec_proj = np.matmul(x_test_proj_vec, self.parameters['second_step'])
        results = neighbor.predict(x_test_proj_vec_proj)
        correct_results = np.where(results == y_test.ravel())[0]
        return len(correct_results) / len(y_test), correct_results

    def Compute_Ratio(self, x_train, y_train):
        Total_centered = DRA.TotalCentered(x_train)
        Between_centered = DRA.BetweenGroupMeanCentered(x_train, y_train)
        _, S, _ = np.linalg.svd(Total_centered, full_matrices=False)
        denominator = np.sum(S)
        _, S, _ = np.linalg.svd(Between_centered, full_matrices=False)
        numerator = np.sum(S)
        return numerator/denominator
