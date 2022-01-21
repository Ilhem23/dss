import numpy as np
import dataloader as dt
import pandas as pd
import gui

def ahp_weight(matrix):
    n= matrix.shape[0]
    m= matrix.shape[1]
    matrixResult = np.zeros((n, m))
    for i in range(0, m):
        matrixResult[:, i] = matrix[:, i] / np.sum(matrix[:, i])
    avr_row= np.mean(matrixResult, axis=1)
    return avr_row

print("Select weight matrix Location \n")
decisionMatrix = dt.loadCsv(gui.openfile("DataFile"))
decisionMatrix= np.array(decisionMatrix)

ahp= ahp_weight(decisionMatrix)
print("##################### AHP Criteria Weight ################### \n")
print(ahp)


