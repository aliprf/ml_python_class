import configuration.config as config
import configuration.sharedData as shd
from statistics import mean
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import math


def calculateDistance(vectorA , vectorB): # for each items: ( (b1-a1)p2 + (b2-a2)p2 + ... ) power 1/2
    sumSquare = 0
    for i in range(len(vectorA)):
        sumSquare += math.pow(abs((vectorB[i] - vectorA[i])), 2)
    x = math.sqrt(sumSquare)
    return x


def calculatePreDiscriminantParameters(index):
    prior = 1 / len(shd.TrainingDataSet.labelIndexes[index])

    meanVector = np.array(shd.TrainingDataSet.meanVectorByLabel.get(index))[np.newaxis]
    tr_meanVector = meanVector.T

    covMat = np.array(shd.TrainingDataSet.covarianceMatrixByLabel[index])

    mInverse = np.linalg.pinv(covMat)
    # detCov = np.linalg.det(shd.TrainingDataSet.covarianceMatrixByLabel[index])
    detCov = np.exp(np.trace(covMat))
    shd.TrainingDataSet.W_matrix_ByLabel[index] = -1 / 2 * mInverse

    tmp = np.dot(mInverse, tr_meanVector)
    shd.TrainingDataSet.w_matrix_ByLabel[index] = tmp.T

    shd.TrainingDataSet.w_zero_ByLabel[index] = -1 / 2 * np.dot(np.dot(meanVector, mInverse), tr_meanVector) \
                                                  - 1 / 2 * np.log(detCov) + np.log(prior)


def calculateEigenValue_and_EigenVector(matrix):
    matrix = np.array(matrix)
    eigenValues, eigenVectors = np.linalg.eig(matrix)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    return eigenValues, eigenVectors


def normalizeDsByMinusToMean(meanVector):

    vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE

    for index in  range(len(shd.TrainingDataSet.imagesDataSet)):
        # retrieve img vector
        vector = shd.TrainingDataSet.imagesDataSet[index]
        normalizedVector = [0] * vectorSize

        for i in range(vectorSize):
            normalizedVector[i] = (vector[i] - meanVector[i])

        shd.TrainingDataSet.imagesDataSet[index] = normalizedVector

        # lv= np.array(vector)
        # mv = np.array(meanVector)
        # nw = np.array(normalizedVector)


def calculateMeanByIndexes(indexes, needTest, dynamicSize):
    if dynamicSize:
        vectorSize = config.MNIST_DS_SAMPLE_REDUCED_IMAGE_SIZE
    else:
        vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE

    meanVector = [0] * vectorSize

    # just for testing
    classDataMatrix = []

    for index in indexes:
        # retrieve img vector
        vector = shd.TrainingDataSet.imagesDataSet[index]

        # calculate mean till this index
        for i in range(vectorSize):
            meanVector[i] = (meanVector[i] + vector[i])

        for i in range(vectorSize):
            meanVector[i] = meanVector[i] / vectorSize

        # just testing it
        # if needTest:
        #     classDataMatrix.append(vector)
        #     # print(vector[:])

    # if needTest:
    #     testMean(classDataMatrix=classDataMatrix, calculatedMean=meanVector)
    x= np.array(meanVector)
    return meanVector


def calculateMeanVector():
    vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE
    meanVector = [0] * vectorSize

    for index in range(len(shd.TrainingDataSet.imagesDataSet)):
        # retrieve img vector
        vector = shd.TrainingDataSet.imagesDataSet[index]

        # calculate mean till this index
        for i in range(vectorSize):
            meanVector[i] = (meanVector[i] + vector[i])

        for i in range(vectorSize):
            meanVector[i] = meanVector[i] / vectorSize

    return meanVector


def calculateCovarianceByIndexes(meanVector, indexes, dynamicSize):  # Sum( (x-m)t(x-m) ) / n
    if dynamicSize:
        vectorSize = config.MNIST_DS_SAMPLE_REDUCED_IMAGE_SIZE
    else:
        vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE

    xMinMeanVector = [0] * vectorSize

    covarianceMatrixDataSet = [[0 for x in range(vectorSize)] for y in range(vectorSize)]
    print('sample_size')
    print(len(indexes))
    for counter, index in tqdm(enumerate(indexes)):  # each sample

        # vector = shd.TrainingDataSet.imagesDataSet[index]
        for i in range(vectorSize):
            xMinMeanVector[i] = shd.TrainingDataSet.imagesDataSet[index][i] - meanVector[i]

        # calculating covarianceMatrix for sample Xi
        covarianceMatrixSamples = calculateCoMatrix(xMinMeanVector)
        for i in range(vectorSize):
            for j in range(vectorSize):
                covarianceMatrixDataSet[i][j] += covarianceMatrixSamples[i][j]

    covMat = np.array(covarianceMatrixDataSet)

    for i in range(vectorSize):
        for j in range(vectorSize):
            # x = random.uniform(0.00000000001, 0.00000000003)
            # covarianceMatrixDataSet[i][j] = (covarianceMatrixDataSet[i][j] + x) / len(indexes)
            covarianceMatrixDataSet[i][j] = (covarianceMatrixDataSet[i][j] ) / len(indexes)

    covMat = np.array(covarianceMatrixDataSet)
    return covarianceMatrixDataSet


def calculateCovariance(meanVector):  # Sum( (x-m)t(x-m) ) / n
    vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE
    xMinMeanVector = [0] * vectorSize

    covarianceMatrixDataSet = [[0 for x in range(vectorSize)] for y in range(vectorSize)]

    for counter, index in tqdm(enumerate(shd.TrainingDataSet.imagesDataSet)):  # each sample

        # vector = shd.TrainingDataSet.imagesDataSet[index]
        for i in range(vectorSize):
            xMinMeanVector[i] = shd.TrainingDataSet.imagesDataSet[counter][i] - meanVector[i]

        # calculating covarianceMatrix for sample Xi
        covarianceMatrixSamples = calculateCoMatrix(xMinMeanVector)
        for i in range(vectorSize):
            for j in range(vectorSize):
                covarianceMatrixDataSet[i][j] += covarianceMatrixSamples[i][j]

    # covMat = np.array(covarianceMatrixDataSet)

    for i in range(vectorSize):
        for j in range(vectorSize):
            covarianceMatrixDataSet[i][j] = (covarianceMatrixDataSet[i][j] ) / len(shd.TrainingDataSet.imagesDataSet)

    covMat = np.array(covarianceMatrixDataSet)
    return covarianceMatrixDataSet


def calculateCoMatrix(vector):  # input is [a, b ,c] --> we calculate xt * x = matrix

    vector = np.array(vector)[np.newaxis]
    tr_vector = vector.T

    covMatrix = np.dot(tr_vector, vector)

    return covMatrix

# def __calculateCovarianceByIndexes(meanVector, indexes):  # Sum( (x-m)t(x-m) ) / n
#     vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE
#     xMinMeanVector = [0] * vectorSize
#
#     covarianceMatrixDataSet = [[0 for x in range(vectorSize)] for y in range(vectorSize)]
#     print('sample_size')
#     print(len(indexes))
#     for counter, index in tqdm(enumerate(indexes)):  # each sample
#
#         # vector = shd.TrainingDataSet.imagesDataSet[index]
#         for i in range(vectorSize):
#             xMinMeanVector[i] = shd.TrainingDataSet.imagesDataSet[index][i] - meanVector[i]
#
#         # calculating covarianceMatrix for sample Xi
#         covarianceMatrixSamples = calculateCoMatrix(xMinMeanVector)
#         for i in range(vectorSize):
#             for j in range(vectorSize):
#                 covarianceMatrixDataSet[i][j] += covarianceMatrixSamples[i][j]
#
#     covMat = np.array(covarianceMatrixDataSet)
#
#     for i in range(vectorSize):
#         for j in range(vectorSize):
#             x = random.uniform(0.00000000001, 0.00000000003)
#             covarianceMatrixDataSet[i][j] = (covarianceMatrixDataSet[i][j] + x) / len(indexes)
#             # covarianceMatrixDataSet[i][j] = (covarianceMatrixDataSet[i][j] ) / len(indexes)
#
#     covMat = np.array(covarianceMatrixDataSet)
#     return covarianceMatrixDataSet

# def _calculateCoMatrix(vector):  # input is [a, b ,c] --> we calculate xt * x = matrix
#     covMatrix = [0] * len(vector)
#
#     for row in range(len(vector)):
#         rowArr = [0] * len(vector)
#         for col in range(len(vector)):
#             rowArr[col] = vector[row] * vector[col]
#         covMatrix[row] = rowArr
#
#     return covMatrix


# def calculateCovarianceByIndexesTest(meanVector, indexes, needTest):
#     vectorSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE * config.MNIST_DS_SAMPLE_IMAGE_SIZE
#     classDataMatrix = []
#
#     for index in indexes:  # each sample x0 ---> xn
#         # retrieve img vector
#         vector = shd.TrainingDataSet.imagesDataSet[index]
#         classDataMatrix.append(vector)
#
#     arr = np.array(classDataMatrix)
#     covMat = np.cov(arr)
#
#     # print(covMat)
#     # print('------------')
#     return covMat


# def testMean(classDataMatrix, calculatedMean):
#     print('testing mean: ')
#     npArr = np.array(classDataMatrix)
#
#     newMean = list(map(mean, zip(calculatedMean)))
#
#     npMean = npArr.mean(axis=0)
#
#     snp = 0
#     sus = 0
#     nus = 0
#
#     for i in range(len(calculatedMean)):
#         snp += npMean[i]
#         nus += newMean[i]
#         sus += calculatedMean[i]
#
#     print('me--> ' + str(sus / len(calculatedMean)) + ' --- ' + str(nus / len(calculatedMean)) + ' <--npm')
#     print('==============')

# #
# a = np.array([[1, 3, 4, 2], [3, 3, 0, 2]])
# X_realization1 = [1,2,3]
# X_realization2 = [2,1,8]
# print(np.cov([X_realization1, X_realization2])) # rowvar false, each column is a variable
# print(np.cov([1, 3, 4, 2],[3, 3, 0, 2]))
#
# print('------------------------------------')
# X = np.array([ [0.1, 0.3, 0.4, 0.8, 0.9],
#                [3.2, 2.4, 2.4, 0.1, 5.5],
#                [10., 8.2, 4.3, 2.6, 0.9]
#              ])
# print(np.cov(X))



# covarianceMatrixSamples = []
# v1 = [1, 2, 3]
# v2 = [2, 5, 0]
#
# covarianceMatrixSamples.append(calculateCoMatrix(v1))
# covarianceMatrixSamples.append(calculateCoMatrix(v2))
# print('covarianceMatrixSamples')
# print(covarianceMatrixSamples)
# print('---------------')
#
# cov = [[0 for x in range(3)] for y in range(3)]
# # print(cov)
# for row in range(2):
#     tmp=covarianceMatrixSamples[row]
#     print(tmp)
#     for i in range(3):
#         for j in range(3):
#             cov[i][j] += tmp[i][j]
#
# print('-----cov----------')
# print(cov)
# print('---------------')