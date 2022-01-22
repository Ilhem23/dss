import numpy as np

def entropy(matrix):
    n= matrix.shape[0]
    m= matrix.shape[1]
    k= 1 / np.log(n)
    matrixResult = np.zeros((n, m))

    for i in range(0,m):
        matrixResult[:,i]= matrix[:,i] / np.sum(matrix[:,i])
    epsilon= 0.00000000001
    matrixResult2= matrixResult * np.log(matrixResult+epsilon)
    sumNormalized= np.sum(matrixResult2, axis= 0)
    entropy= 1 - (-k * sumNormalized)
    entropy= entropy / np.sum(entropy)
    return entropy
