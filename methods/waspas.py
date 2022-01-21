import numpy as np
import dataloader as dt
import pandas as pd
import gui

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

print("Select Data Location \n")
decisionMatrix = dt.loadCsv(gui.openfile("DataFile"))
decisionMatrix= np.array(decisionMatrix)
print("##################### decision matrix ################### \n")
print(decisionMatrix)
print("Select Weight Location")
weights = np.array(dt.loadCsv(gui.openfile("WeightFile")))
weight= np.eye(decisionMatrix.shape[0], decisionMatrix.shape[1])
weights=np.array(weights * weight)

print("##################### Weight ################### \n")
print(weights)


#criterion_type = ['min', 'max', 'max', 'max','min', 'max', 'max', 'max', 'min', 'max', 'max']
criterion_type = ['max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max']
wsm, wpm, waspass= wsm_wpm_waspass(decisionMatrix, weights, criterion_type, 0.5)
print("##################### WSM Rank ################### \n")
print(wsm)
print("##################### WPM Rank ################### \n")
print(wpm)
print("##################### WASPAS Rank ################### \n")
print(waspass)
#for i in range(0, wsmRank.shape[0]):
#  print('a'+str(i+1), wsmRank[i])

