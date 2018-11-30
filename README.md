# This is a base repository for the other repositories.
In this repository, I store some base codes for the other repositories.


## Classifier
There is a LinearDiscriminantClassifier subclass of Classifier. This subclass takes two 
arguments: discriminant_function and classification_function. The first is used to reduce the 
dimension of the data. And the second is used to classify the reduced data.
The options of the first argument could be the function in the "DimensionReduction" class 
(please refer to https://github.com/randysuen1991/Dimension-Reduction-Approaches/blob/master/DimensionReductionApproaches.py)
and the ones of the second argument could be SVM(http://scikit-learn.org/stable/modules/svm.html), 
KNN(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) ...(to be updated more).

## Two step classifier
It reduce the dimension of the data in two steps, so that the precision of the classification would be 
higher. <br>
For detail, please refer to my master thesis.

## Regressor
There are several subclasses of Regressor class. Most of them are based on sklearn module.

## HaHa
