#!/usr/bin/env python

import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
import configuration.sharedData as sharedData
import utils.mathHelper as mh
import utils.image_printer as imgPrinter
import utils.base_utils.colors as colors
import numpy as np
from tqdm import tqdm
import configuration.config as config
from collections import defaultdict
import classification_algorithms.beysianClassifier as bc
import classification_algorithms.knnClassifier_thread as KNN
import classification_algorithms.svm as svm


def reduceDimensionAndCreateNewData(accuracyIndex):
    PCAmatrix = sharedData.TrainingDataSet.PCAs.get(accuracyIndex)
    _len_train = len(sharedData.TrainingDataSet.imagesDataSet)
    _len_test = len(sharedData.TestSetData.imagesDataSet)

    # for all imgData with datasetLabel, calculate [28*784] * [784*1] => [25*1]
    for index in range(_len_train):
        # retrieve img vector
        vector = np.array(sharedData.TrainingDataSet.imagesDataSet[index]).T # 784 *1
        reducedVector = np.dot(PCAmatrix, vector)
        sharedData.TrainingDataSet.imagesDataSet[index] = reducedVector.tolist()
        print('')

    for index in range(_len_test):
        # retrieve img vector
        vector = np.array(sharedData.TestSetData.imagesDataSet[index]).T # 784 *1
        reducedVector = np.dot(PCAmatrix, vector)
        sharedData.TestSetData.imagesDataSet[index] = reducedVector.tolist()
        print('')


def calculateEigenDiversity(accuracyArray):
    for index in range(len(accuracyArray)):
        accuracyLevel = accuracyArray[index]

        sumTotal =0
        sumByValue = 0
        lenEigenValues = len(sharedData.TrainingDataSet.eigenValue)
        for i in range(lenEigenValues):
            sumTotal += sharedData.TrainingDataSet.eigenValue[i]

        accuracyIndex = 0
        for i in range(lenEigenValues):
            sumByValue += sharedData.TrainingDataSet.eigenValue[i]
            if sumByValue * 100 / sumTotal >= accuracyLevel:
                break
            accuracyIndex += 1

        # now we calculate the PCAs shape is x*784 ==x:25===>  [25*784] * [784*1] => [25*1] : features reduced from
        # 784 to 25
        reducedEigenVectors = sharedData.TrainingDataSet.eigenVector[0:accuracyIndex+1]
        sharedData.TrainingDataSet.PCAs[accuracyLevel] = reducedEigenVectors


def loadData():
    dataLoader.loadTrainingSets(testIndex=-1, fast=False, sampleSize=100, dt_type=config.dataset_type.mnist_hw)
    dataLoader.loadTestSets(dt_type=config.dataset_type.mnist_hw)


def startPCA():
    print(colors.bcolors.HEADER + ' calculating Mean Vector' )
    sharedData.TrainingDataSet.meanVector = mh.calculateMeanVector()

    print(colors.bcolors.HEADER + ' Normalize data by Minus by Mean Vector')
    mh.normalizeDsByMinusToMean(meanVector=sharedData.TrainingDataSet.meanVector)

    print(colors.bcolors.HEADER + ' calculating normalized_data_Mean Vector  ')
    sharedData.TrainingDataSet.meanVector = mh.calculateMeanVector()

    print(colors.bcolors.HEADER + ' calculating CoV Matrix  ')
    sharedData.TrainingDataSet.covarianceMatrix = mh.calculateCovariance(meanVector=sharedData.TrainingDataSet.meanVector)

    print(colors.bcolors.HEADER + ' calculating eigen value & vector : ')
    sharedData.TrainingDataSet.eigenValue, sharedData.TrainingDataSet.eigenVector = \
        mh.calculateEigenValue_and_EigenVector(matrix=sharedData.TrainingDataSet.covarianceMatrix)

    print(colors.bcolors.HEADER + ' calculating All PCAs: ')
    calculateEigenDiversity(config.PCA_ACCURACY_ARRAY)


loadData()
startPCA()

for index in range(len(config.PCA_ACCURACY_ARRAY)):
    print(colors.bcolors.OKGREEN + '<<<<<<<<<<<<<<-------|||------->>>>>>>>>>>>>')
    print(colors.bcolors.OKGREEN + ' reducing dimension with accuracy : '
          + colors.bcolors.OKBLUE + str(config.PCA_ACCURACY_ARRAY[index]) + '%')
    print(colors.bcolors.HEADER + ' Reducing Dimensions and Creating new Data ' )
    reduceDimensionAndCreateNewData(accuracyIndex=config.PCA_ACCURACY_ARRAY[index])

    print('<><><><><> Dimension Reduction Completed: <><><><>')
    x = sharedData.TrainingDataSet.imagesDataSet[1]
    y = np.array(sharedData.TrainingDataSet.imagesDataSet[1])

    # print('KNN Method Started ==>')
    # KNN.startKNN(_dynamicSize=True)
    #
    # print('Beysian Method Started ==>')
    # bc.startBeysian(_dynamicSize=True)

    print('SVM Started ==>')
    svm.thread_svm_init(False)
