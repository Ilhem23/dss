import numpy as np
import pandas as pd

def topsis(matrix, weight, criteria_type, lamda):
    n= matrix.shape[0]
    m= matrix.shape[1]
    matrixResult= np.zeros((n,m))
    matrix2= np.square(matrix)
    for i in range(0,m):
        matrixResult[:, i] = matrix[:, i] / np.sqrt(np.sum(matrix2[:, i]))
    matrixResult= matrixResult * weight
    max_vector= np.max(matrixResult, axis=0)
    min_vector= np.min(matrixResult, axis=0)
    max_vecto= np.zeros(m)
    min_vecto= np.zeros(m)
    for j in range(0,m):
        if criteria_type[j] == "max":
            max_vecto[j]= max_vector[j]
            min_vecto[j]= min_vector[j]
        else:
            max_vecto[j] = min_vector[j]
            min_vecto[j] = max_vector[j]
    s_i_plus= np.sqrt(np.sum(np.square(matrixResult - max_vecto), axis=1) )
    s_i_moins = np.sqrt(np.sum(np.square(matrixResult - min_vecto), axis=1))
    pi= s_i_plus + s_i_moins
    pi= s_i_moins / pi
    df = pd.DataFrame(data=pi, columns=["pi"])
    df["rank"] = df.size - df["pi"].rank(method='min') + 1
    return df



