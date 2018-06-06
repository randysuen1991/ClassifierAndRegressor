import numpy as np
import pandas as pd
import sys
if 'C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches')
from DimensionReductionApproaches import CenteringDecorator



class ModelSelection():
    # Till now, Y_train should be a N*1 matrix.
    @CenteringDecorator
    def CovSelection(X_train,Y_train):
        covariance = np.matmul(X_train.T,Y_train)
        
        
    def StepSelection():
        pass
    
    
    
    
    
    
    