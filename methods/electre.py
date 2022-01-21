import numpy as np
import dataloader as dt
import gui


def normalized_matrix(decisionMatrix):
    decisionMatrix = np.matrix(decisionMatrix)
    sqaureMatrix = np.square(decisionMatrix)
    sumMatrix = sqaureMatrix.sum(axis=0, dtype='float')
    sqrtArray = np.array(np.sqrt(sumMatrix))
    sqrtArray = sqrtArray[0]
    normalizedMatrix = decisionMatrix / sqrtArray[None, :]
    return normalizedMatrix


def weighted_normalize_matrix(normalizedMatrix, weigth):
    weigth = np.array(weigth)
    kernel = np.eye(len(weigth))
    weigthMatrix = kernel * weigth
    weightedNormalizeMatrix = normalizedMatrix * weigthMatrix
    return weightedNormalizeMatrix


def concordance_matrix(weigthedNormalizedMatrix, weigth):
    n = weigthedNormalizedMatrix.shape[0]
    m = weigthedNormalizedMatrix.shape[1]
    concordanceMatrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, m):
            value = 0
            for k in range(0, m):
                if (j != i):
                    ## select the criteria where alternative a is better than b  :  𝑪_𝒂𝒃= {𝒋│𝒙_𝒂𝒋≥ 𝒙_𝒃𝒋 }
                    if (weigthedNormalizedMatrix[i, k] >= weigthedNormalizedMatrix[j, k]):
                        ## Sum of the criteria weights belongs to the interval set calculated
                        value = value + weigth[k]
            concordanceMatrix[i, j] = value
    if (np.sum(weigth) != 0):
        concordanceMatrix = concordanceMatrix / np.sum(weigth)
    return concordanceMatrix

def negative_from_array(arr):
    matr= []
    for k in arr:
        if (k < 0):
            matr.append(k)
    return np.array(matr)


def discordance_matrix(matrix):
    n= matrix.shape[0]
    discordanceMatrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            calculat_difference=np.array(matrix[i, :] - matrix[j, :])
            calculat_differenc= np.array(calculat_difference[0])
            if not any(calculat_differenc):
                discordanceMatrix[i, j]= 0
            else:
                negative_values = negative_from_array(calculat_differenc)
                if negative_values.size == 0:
                    numerator = 0
                else:
                    numerator = max(np.abs(negative_values))
                denominator = max(np.abs(calculat_differenc))
                discor_index= numerator / denominator
                discordanceMatrix[i, j] = discor_index

    return discordanceMatrix

def calcul_thresehold(matrix):
    n= matrix.shape[0]
    result= 0
    for i in range(0,n):
        result += np.sum(matrix[i, :])
    return result / (n * (n - 1))

def concordance_index(matrix, thresehold):
    n= matrix.shape[0]
    matrix_result= np.zeros((n, n))
    for i in range(0,n):
        for j in range(0,n):
            if matrix[i,j] < thresehold:
                matrix_result[i,j]= 0
            else:
                matrix_result[i,j]= 1

    return matrix_result

def discordance_index(matrix, thresehold):
    n= matrix.shape[0]
    matrix_result= np.zeros((n, n))
    for i in range(0,n):
        for j in range(0,n):
            if matrix[i,j] <= thresehold:
                matrix_result[i,j]= 1
            else:
                matrix_result[i,j]= 0

    return matrix_result

def net_calcul(matrix):
    sum_col= np.sum(matrix, axis=0)
    sum_row= np.sum(matrix, axis= 1)
    return sum_row - sum_col
def net_sup_rank(matrix):
    net_sup_rank = np.argsort(matrix)
    array_result= []
    for value in net_sup_rank:
        val= "A" + str(value + 1)
        array_result.append(val)
    n= len(array_result)
    return array_result[::-1][:n]

def net_inf_rank(matrix):
    net_sup_rank = np.argsort(matrix)
    array_result= []
    for value in net_sup_rank:
        val= "A" + str(value + 1)
        array_result.append(val)
    return array_result


print("Select Data Location")
decisionMatrix = dt.loadCsv(gui.openfile("DataFile"))
decisionMatrix= np.matrix(decisionMatrix)
print(decisionMatrix)
print("Select Weight Location")
weigth = dt.loadCsv(gui.openfile("WeightFile"))
weigth= np.array(weigth)
weigth= weigth[0]
print(weigth)
#decisionMatrix = [[1350, 1850, 44323], [1680, 1650, 44324], [1560, 1950, 44322]]
#weigth = [0.3357, 0.2076, 0.4567]
normalizeMatrix = normalized_matrix(decisionMatrix)
weigthedNormalizedMatrix = weighted_normalize_matrix(normalizeMatrix, weigth)
print("/////// Normalized Matrix ////////// \n ")
print(normalizeMatrix)
print("\n /////// weigthed Normalized Matrix /////// \n ")
print(weigthedNormalizedMatrix)
print("\n /////// Concordance Matrix /////// \n ")
concordanceMatrix = concordance_matrix(weigthedNormalizedMatrix, weigth)
print(concordanceMatrix)
print("\n /////// Discordance Matrix /////// \n ")
discordanceMatrix = discordance_matrix(weigthedNormalizedMatrix)
print(discordanceMatrix)
print("\n /////// concordance thresehold /////// \n ")
c_bar= calcul_thresehold(concordanceMatrix)
print(c_bar)
print("\n /////// discordance thresehold /////// \n ")
d_bar= calcul_thresehold(discordanceMatrix)
print(d_bar)
print("\n /////// concordance index matrix /////// \n ")
concordanceIndexMatrix = concordance_index(concordanceMatrix, c_bar)
print(concordanceIndexMatrix)
print("\n /////// discordance index matrix /////// \n ")
discordanceIndexMatrix = discordance_index(discordanceMatrix, d_bar)
print(discordanceIndexMatrix)
print("\n /////// net superior /////// \n ")
net_sup= net_calcul(concordanceMatrix)
print(net_sup)
print("\n /////// net inferior /////// \n ")
net_inf= net_calcul(discordanceMatrix)
print(net_inf)

print("\n /////// rank net superior /////// \n ")
net_sup_rank= net_sup_rank(net_sup)
print(net_sup_rank)


print("\n /////// rank net inferior /////// \n ")
net_inf_rank= net_inf_rank(net_inf)
print(net_inf_rank)