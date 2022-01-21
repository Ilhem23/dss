import numpy as np
import pandas as pd


def wsm_wpm_waspass(matrix, weight, criteria_type, lamda):
    n= matrix.shape[0]
    m= matrix.shape[1]
    matrixResult= np.zeros((n,m))
    for i in range(0, m):
        if criteria_type[i] == 'max':
            if np.max(matrix[:,i])== 0:
                matrixResult[:, i]= 0
            else:
                matrixResult[:, i] = np.divide(matrix[:, i], np.max(matrix[:, i]))
        else:
            if np.min(matrix[:, i]) == 0:
                matrixResult[:, i] = 0
            else:
                matrixResult[:, i] = np.divide(np.min(matrix[:, i]), matrix[:, i])
    matrixResult1= matrixResult * weight
    matrixResult2 = matrixResult ** weight
    matrixWSM= np.sum(matrixResult1, axis=1)
    matrixWpm= np.prod(matrixResult2, axis=1)
    print(matrixWpm)
    waspas = (lamda * matrixWSM )+ ((1 - lamda) * matrixWpm)
    df = pd.DataFrame(data=matrixWSM, columns=["WSM"])
    df["rank"] = df.size - df["WSM"].rank(method='min') + 1
    df2 = pd.DataFrame(data=matrixWpm, columns=["WPM"])
    df2["rank"] = df2.size - df2["WPM"].rank(method='min') + 1
    df3 = pd.DataFrame(data=waspas, columns=["WASPAS"])
    df3["rank"] = df3.size - df3["WASPAS"].rank(method='min') + 1
    return df, df2, df3
