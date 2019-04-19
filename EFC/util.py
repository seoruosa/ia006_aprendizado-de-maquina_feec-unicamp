# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def series_to_supervised(data_array, K=1):
    X = np.array([])
    Y = np.array([])

    for i in range(len(data_array)-K):
        X = np.append(X, data_array[i:K+i])
        Y = np.append(Y, data_array[K+i])
    X = X.reshape((-1,K))
    return (X, Y)

