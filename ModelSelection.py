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
    def CorrSelection(X_train,Y_train,both_sides,num_each_side,abs=False,**kwargs):
        if Y_train.shape[1] > 1 :
            warnings.warn('The dimension of the Y variable should be 1 now.')
        covariance = np.matmul(X_train.T,Y_train)/X_train.shape[0]
        X_std = np.expand_dims(np.std(X_train,axis=0),axis=1)
        Y_std = np.std(Y_train)
        corr = covariance/(X_std*Y_std)
        
        if abs == True :
            if num_each_side == 'full' :
                num_each_side = X_train.shape[1]
                
            corr = np.abs(corr)
            sorted_index_all = np.argsort(corr.ravel())
            sorted_index_all = sorted_index_all[::-1]
            sorted_left = sorted_index_all[0:num_each_side]
            sorted_index = sorted_left
            if both_sides == True:
                warnings.warn('You should not have picked the both sides of the variable list.')
            return sorted_index, np.reshape(corr[sorted_index],newshape=(1,len(corr[sorted_index])))
        
        sorted_index_all = np.argsort(corr.ravel())
        sorted_index_all = sorted_index_all[::-1]
        
        if both_sides == True :
            
            # if fill, it will dsciard the weakest one
            if num_each_side == 'full' :
                num_each_side = np.floor(X_train.shape[1]/2)
                
            index_left = sorted_index_all[0:num_each_side]
            index_right = sorted_index_all[-1:-1-num_each_side:-1]
            sorted_index = np.concatenate([index_left,index_right],axis=0)
        else:
            if num_each_side == 'full' :
                num_each_side = X_train.shape[1]
                
            index_left = sorted_index_all[0:num_each_side]
            sorted_index = index_left
            
        return sorted_index, np.reshape(corr[sorted_index],newshape=(1,len(corr[sorted_index])))
        
    def FowardSelection():
        pass
    
    def BackwardSelection():
        pass
    
    def StepwiseSelection():
        pass
    
    
    
    
    
    
    