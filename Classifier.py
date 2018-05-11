import sys
if 'C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches')
import UtilFun as UF


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
        self.parameters = self.discriminant_function(X_train=X_train,Y_train=Y_train,kwargs=self.kwargs)
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
        p_tilde = kwargs.get('p_tilde')
        q_tilde = kwargs.get('q_tilde')
        self.input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
        self._Search_Dimensionality(X_train=X_train,Y_train=Y_train)
        X_train_proj = np.matmul(X_train,self.parameters['first_step']['row'])
        X_train_proj = np.matmul(self.parameters['first_step']['column'],X_train_proj)
        X_train_proj_vec = UF.imgs2vectors(X_train_proj)
        self.transformed_X_train = np.matmul(X_train_proj_vec,self.parameters['second_step'])
        return self.parameters
    
    def _Search_Dimensionality(self,X_train,Y_train,p_tilde,q_tilde,dimension=50):
        
        for iter in range(dimension):
            linear_subspace = self.first_step_function(X_train = X_train,input_shape = self.input_shape,p_tilde=p_tilde,q_tilde=q_tilde)
            X_train_proj = np.matmul(np.matmul(linear_subspace[0],X_train),linear_subspace[1])
            
            
            
        self.parameters['first_step'] = 
        self.parameters['second_step'] = 
    
    
    def Classify(self,X_train,Y_train,X_test,Y_test,**kwargs):
        k = kwargs.get('k',1)
        # Use K-nearest neighbor to classify the testing data
        neighbor = KNeighborsClassifier(n_neighbors=k)
        neighbor.fit(self.transformed_X_train,self.Y_train.ravel())
        
        results = neighbor.predict(X_test_proj)
        correct_results = np.where(results == Y_test.ravel())[0]
        return len(correct_results) / len(Y_test), correct_results

            