import numpy as np
import pandas as pd
import sys
if 'C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches')
from DimensionReductionApproaches import CenteringDecorator
import warnings


class ModelSelection():
    # Till now, Y_train should be a N*1 matrix.
    @CenteringDecorator
    def CovSelection(X_train,Y_train):
        if Y_train.shape[1] > 1 :
            warnings.warn('The dimension of the Y variable should be 1 now.')
        covariance = np.matmul(X_train.T,Y_train)
        sorted_index = np.argsort(covariance)
        return sorted_index[::-1]
        
    def StepSelection():
        pass
    
    
    
    
    
    
    