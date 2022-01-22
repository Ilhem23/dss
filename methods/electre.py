import numpy as np


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


def net_sup(matrix):
    net_sup_rank = np.argsort(matrix)
    array_result= []
    for value in net_sup_rank:
        val= "A" + str(value + 1)
        array_result.append(val)
    n= len(array_result)
    return array_result[::-1][:n]

def net_inf(matrix):
    net_sup_rank = np.argsort(matrix)
    array_result= []
    for value in net_sup_rank:
        val= "A" + str(value + 1)
        array_result.append(val)
    return array_result


def electre(matrix, thresehold, weigth):

    normalizeMatrix = normalized_matrix(matrix)
    weigthedNormalizedMatrix = weighted_normalize_matrix(normalizeMatrix, weigth)
    concordanceMatrix = concordance_matrix(weigthedNormalizedMatrix, weigth)
    discordanceMatrix = discordance_matrix(weigthedNormalizedMatrix)
    c_bar = calcul_thresehold(concordanceMatrix)
    d_bar = calcul_thresehold(discordanceMatrix)
    net_su = net_calcul(concordanceMatrix)
    net_in = net_calcul(discordanceMatrix)
    net_sup_rank = net_sup(net_su)
    net_inf_rank = net_inf(net_in)

    return net_sup_rank, net_inf_rank


