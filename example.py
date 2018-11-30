import numpy as np
import matplotlib.pyplot as plt

# In this example, I demonstrate different regressions methods. 
def example1():
    
    noise = np.random.normal(loc=0, scale=10, size=(100,))
    X_train = np.random.randint(low=1, high=20, size=(100,))
    Y_train = 1 + 2 * X_train + noise
    a = 5
    b = 4
    c = 6
    d = 5



if __name__ == '__main__' :
    example1()