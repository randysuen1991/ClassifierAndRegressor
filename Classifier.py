import numpy as np

from sklearn.neighbors import KNeighborsClassifier
        

    
class Classifier():
    def __init__(self):
        self.parameters = None
        self.transformed_X_train = None
        self.Y_train = None
        self.kwargs = dict()
    def Fit(cls):
        raise NotImplementedError
    def Classify(cls):
        raise NotImplementedError
            
class LinearDiscriminantClassifier(Classifier):
    
    def __init__(self,discriminant_function,**kwargs):
        super().__init__()
        self.discriminant_function = discriminant_function
        self.kwargs = kwargs
        
    def Fit(self,X_train,Y_train):
        self.parameters = self.discriminant_function(X_train,Y_train,kwargs=self.kwargs)
        return self.parameters
    
    def Classify(self,X_train,Y_train,X_test,Y_test):
        k = self.kwargs.get('k',1)
        X_train_proj = np.matmul(X_train,self.parameters)
        X_test_proj = np.matmul(X_test,self.parameters)
        # Use K-nearest neighbor to classify the testing data
        neighbor = KNeighborsClassifier(n_neighbors=k)
        neighbor.fit(X_train_proj,Y_train.ravel())
        results = neighbor.predict(X_test_proj)
        correct_results = np.where(results == Y_test.ravel())[0]
        return len(correct_results) / len(Y_test), correct_results
        
class TwoStepClassifier(Classifier):
    def __init__(self,first_step_function,second_step_function,**kwargs):
        super().__init__()
        self.first_step_function = first_step_function
        self.second_step_classifier = LinearDiscriminantClassifier(discriminant_function=second_step_function)
        self.parameters = dict()
        
    
    def Fit(self,X_train,Y_train,**kwargs):
        self.parameters['first_step'] = self.first_step_function(X_train,Y_train,kwargs)
        X_train_proj = np.matmul(X_train,self.parameters['first_step'])
        self.parameters['second_step'] = self.second_step_classifier.Fit(X_train_proj,Y_train,kwargs)
        self.transformed_train = np.matmul(X_train_proj,self.parameters['second_step'])
        return self.parameters
    def Classify(self,X_train,Y_train,X_test,Y_test,**kwargs):
        k = kwargs.get('k',1)
        # Use K-nearest neighbor to classify the testing data
        neighbor = KNeighborsClassifier(n_neighbors=k)
        neighbor.fit(self.transformed_X_train,self.Y_train)
        

            