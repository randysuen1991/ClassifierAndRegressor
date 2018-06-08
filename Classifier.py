import sys
if 'C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches')
import UtilFun as UF
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import DimensionReductionApproaches as DRA
import warnings

# This decorator would identify the classifier. This should decorate the Fit function of the 
# Classifier. 
def ClassifierDecorator():
    def decofun():
        
        return None
    
    return decofun
    


# I should have written a linear discriminant decoarator to decorate the method of LinearDiscriminantClassifier.
# i.e it oculd project the input argument into the discriminitive subspace.


# I also should have written a decorator for 'self.Classsify'. Since sometimes we don't have the labels of testing data.

# This is a failure decorator. I want to use it to decorate a class function, but it will miss 'self' argument.
    




class ClassifyDeco():
    def __init__(self,f):
        self.f = f
    def __call__(self,**kwargs):
        print(self.f)
        results = self.f(X_test = kwargs.get('X_test'))
        Y_test = kwargs.get('Y_test',None)
        correct_results = np.where(results == Y_test.ravel())[0]
        if type(Y_test) == np.array :
            return len(correct_results) / len(Y_test), results
        else :
            return results



"""
Notice :
The labels of the data, i.e the Y should always be numeric number larger or equal to 1
"""
"""
Notice : 
I should add some new classifier like random forest classifier, decision tree, bayes, nueral network classifier, adaboost classifier...
"""



class Classifier():
    def __init__(self,classify_fun=None,**kwargs):
        self.parameters = None
        self.transformed_X_train = None
        self.Y_train = None
        self.kwargs = dict()
        self.classify_fun = classify_fun
        
    
    def Fit(self,X_train,Y_train):
        if self.classify_fun == None :
            raise NotImplementedError('Please pass a classify function before fitting the classifier.')
        if self.classify_fun == KNeighborsClassifier :
            self.classifier = self.classify_fun(self.kwargs.get('k',1))
        elif self.classify_fun  in (SVC, AdaBoostClassifier, RandomForestClassifier, GaussianNB,
                                        DecisionTreeClassifier, GaussianProcessClassifier,
                                        QuadraticDiscriminantAnalysis,  MultinomialNB,
                                        BernoulliNB ):
            if self.classify_fun == MultinomialNB or self.classify_fun == BernoulliNB :
                warnings.warn('Please notice that the explanatory variables should be discrete data.')
            self.classifier = self.classify_fun()
        
        self.classifier.fit(X_train,Y_train.ravel())
        
    def Classify(self,X_test,Y_test = None):
        results = self.classifier.predict(X_test)
        
        
        if type(Y_test) != np.ndarray :
            return results
        else :
            correct_results = np.where(results == Y_test.ravel())[0]
            return len(correct_results) / len(Y_test), results, correct_results

class LinearDiscriminantClassifier(Classifier):
    
    def __init__(self,discriminant_fun,classify_fun,**kwargs):
        super().__init__()
        self.discriminant_function = discriminant_fun
        self.classify_function = classify_fun
        self.kwargs = kwargs
    
    
    def Fit(self,X_train,Y_train):
        self.parameters = self.discriminant_function(X_train=X_train,Y_train=Y_train,kwargs=self.kwargs)
        X_train_proj = np.matmul(X_train,self.parameters)
        if self.classify_function == KNeighborsClassifier :
            self.classifier = self.classify_function(self.kwargs.get('k',1))
        elif self.classify_function in (SVC, AdaBoostClassifier, RandomForestClassifier, GaussianNB,
                                        DecisionTreeClassifier, GaussianProcessClassifier,
                                        QuadraticDiscriminantAnalysis, MultinomialNB,
                                        BernoulliNB ):
            if self.classify_fun == MultinomialNB or self.classify_fun == BernoulliNB :
                warnings.warn('Please notice that the explanatory variables should be discrete data.')
            self.classifier = self.classify_function()
        self.classifier.fit(X_train_proj,Y_train.ravel())
        
        return self.parameters
    

    def Classify(self,X_test,Y_test=None):
        X_test_proj = np.matmul(X_test,self.parameters)
        results = self.classifier.predict(X_test_proj)
        if type(Y_test) != np.ndarray :
            return results
        else :
            correct_results = np.where(results == Y_test.ravel())[0]
            return len(correct_results) / len(Y_test), results, correct_results
        
class TwoStepClassifier(Classifier):
    
    def __init__(self,first_step_function,second_step_function,classify_function,**kwargs):
        super().__init__()
        self.first_step_function = first_step_function
        self.second_step_function = second_step_function
        self.classify_function = classify_function
        self.parameters = dict()
        
    
    def Fit(self,X_train,Y_train,**kwargs):

        self.input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
        dimension = self._Search_Dimensionality(X_train = X_train,
                                                Y_train = Y_train)
        linear_subspace = self.first_step_function(X_train = X_train,
                                                   Y_train = Y_train,
                                                   p_tilde = dimension,
                                                   q_tilde = dimension)
        
        self.parameters['first_step']['row'] = linear_subspace[0]
        self.parameters['first_step']['column'] = linear_subspace[1]
        
        X_train_proj = DRA.MultilinearReduction.TensorProject(X_train,self.parameters['first_step']['row'],self.parameters['first_step']['column'])
        X_train_proj_vec = UF.imgs2vectors(X_train_proj)
        
        self.parameters['second_step'] = self.second_step_function(X_train=X_train,Y_train=Y_train)
        
        self.transformed_X_train = np.matmul(X_train_proj_vec,self.parameters['second_step'])
        self.Y_train = Y_train
        return self.parameters
    
    def _Search_Dimensionality(self,X_train,Y_train,p_tilde,q_tilde,dimension=50):
        
        
        ratios = []
        for iter in range(dimension):
            linear_subspace = self.first_step_function(X_train = X_train,input_shape = self.input_shape,p_tilde=iter+1,q_tilde=iter+1)
            X_train_proj = np.matmul(np.matmul(linear_subspace[0],X_train),linear_subspace[1])
            X_train_proj_vec = UF.imgs2vectors(X_train_proj)
            linear_subspace = self.second_step_function(X_train_proj,Y_train)
            X_train_proj_vec_proj = np.matmul(X_train_proj_vec,linear_subspace)
            ratio = self.Compute_Ratio(X_train_proj_vec_proj,Y_train)
            ratios.append(ratio)
        
        ratios = np.round(ratios,6)
        index = np.argmax(ratios)
        
        return index + 1
    
    
    def Classify(self,X_test,Y_test,**kwargs):
        k = kwargs.get('k',1)
        # Use K-nearest neighbor to classify the testing data
        neighbor = KNeighborsClassifier(n_neighbors=k)
        neighbor.fit(self.transformed_X_train,self.Y_train.ravel())
        
        X_test_proj = DRA.MultilinearReduction.TensorProject(X_test,self.parameters['first_step']['row'],self.parameters['first_step']['column'])
        X_test_proj_vec = UF.imgs2vectors(X_test_proj)
        X_test_proj_vec_proj = np.matmul(X_test_proj_vec,self.parameters['second_step'])
        results = neighbor.predict(X_test_proj_vec_proj)
        correct_results = np.where(results == Y_test.ravel())[0]
        return len(correct_results) / len(Y_test), correct_results

    def Compute_Ratio(self,X_train,Y_train):
        Total_centered = DRA.TotalCentered(X_train)
        Between_centered = DRA.BetweenGroupMeanCentered(X_train,Y_train)
        _, S, _ = np.linalg.svd(Total_centered,full_matrices=False)
        denominator = np.sum(S)
        _, S, _ = np.linalg.svd(Between_centered,full_matrices=False)
        numerator = np.sum(S)
        return numerator/denominator
    