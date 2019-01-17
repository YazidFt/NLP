import numpy as np
import pandas as pd

#Processing the data

def get_data():
    
    data = pd.read_csv('ecommerce_data.csv')
    data = data.as_matrix()

    X = data[:,:-1]
    Y = data[:,-1]
    X[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean())/X[:,2].std()

    #One hot-encoding the time_of_day variable
    N, D = X.shape
    #The new Data
    # D+3 because we have for categories {0,1,2,3}
    X2 = np.zeros((N,D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]

    for i in range(N):
        idx = int(X2[i,D-1])
        X2[i, idx+D-1] = 1

    return X2, Y


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <=1]
    Y2 = X[Y <=1]
    return X2, Y2