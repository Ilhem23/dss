import numpy as np
import dataloader as dt
import pandas as pd
import gui

def entropy(matrix):
    n= matrix.shape[0]
    m= matrix.shape[1]
    k= 1 / np.log(n)
    matrixResult = np.zeros((n, m))
    for i in range(0,m):
        matrixResult[:, i]= matrix[:,i] / np.sum(matrix[:, i])
    matrixResult= matrixResult * np.log(matrixResult)
    sumNormalized= np.sum(matrixResult, axis= 0)
    entropy= 1 - (-k * sumNormalized)
    entropy= entropy / np.sum(entropy)
    return entropy

print("Select Data Location \n")
decisionMatrix = dt.loadCsv(gui.openfile("DataFile"))
decisionMatrix= np.array(decisionMatrix)
print("##################### decision matrix ################### \n")
print(decisionMatrix)

entropy_weight= entropy(decisionMatrix)
print("##################### Entropy weight ################### \n")
print(entropy_weight)

