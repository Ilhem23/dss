import numpy as np
import pandas as pd

def ahp_weight(matrix):
    n= matrix.shape[0]
    m= matrix.shape[1]
    matrixResult = np.zeros((n, m))
    for i in range(0, m):
        matrixResult[:, i] = matrix[:, i] / np.sum(matrix[:, i])
    avr_row= np.mean(matrixResult, axis=1)
    return avr_row
