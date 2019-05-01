import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
import configuration.sharedData as sharedData
import utils.mathHelper as mh
import utils.image_printer as imgPrinter
import utils.base_utils.colors as colors
import numpy as np
from tqdm import tqdm
import configuration.config as config


def calculateDiscriminant(inputVector, label):  # the input is array, so we consider it as a transpose of the vector
    inputVector = np.array(inputVector)[np.newaxis]
    tr_inputVector = inputVector.T

    discArr = []
    discArrLbl = []

    for i, values in sharedData.TrainingDataSet.labelIndexes.items():

        gx1 = np.dot(np.dot(inputVector, sharedData.TrainingDataSet.W_matrix_ByLabel[i]), tr_inputVector)
        gx2 = np.dot(sharedData.TrainingDataSet.w_matrix_ByLabel[i], tr_inputVector)
        gx3 = sharedData.TrainingDataSet.w_zero_ByLabel[i]

        discArr.append(gx1[0][0] + gx2[0][0] + gx3[0][0])
        discArrLbl.append(i)

    maximum_gx = max(discArr)
    predicted_lbl = discArrLbl[discArr.index(maximum_gx)]

    #
    # print("correct_label:" + str(label))
    # print("predicted_label:" + str(predicted_lbl))

    sharedData.Results.number_of_all_samples += 1
    if label == predicted_lbl:
        sharedData.Results.number_of_corrects += 1

    # create consusion Matrix
    rslArr = sharedData.Results.resultByIndex[label]
    if rslArr is None or len(rslArr) == 0:
        rslArr = [0] * 10
        sharedData.Results.resultByIndex[label] = rslArr
    else:
        rslArr[predicted_lbl] += 1


def loadData():
    dataLoader.loadTrainingSets(testIndex=-1, fast=False, sampleSize=5000, dt_type=config.dataset_type.mnist_hw)
    dataLoader.loadTestSets(dt_type=config.dataset_type.mnist_hw)


def startBeysian(_dynamicSize):
    # calculating mean and covariance for each dataset

    if _dynamicSize:
        config.MNIST_DS_SAMPLE_REDUCED_IMAGE_SIZE = len(sharedData.TrainingDataSet.imagesDataSet[0])

    x = config.MNIST_DS_SAMPLE_IMAGE_SIZE
    print()
    for i, values in sharedData.TrainingDataSet.labelIndexes.items():

        print(colors.bcolors.HEADER + ' calculating Mean Vector for Dataset_Label: ' + colors.bcolors.OKBLUE + str(i))
        sharedData.TrainingDataSet.meanVectorByLabel[i] = mh.calculateMeanByIndexes(indexes=values, needTest=False,
                                                                                    dynamicSize=_dynamicSize)

        meanVectorByLabel = np.array(sharedData.TrainingDataSet.meanVectorByLabel[i])
        print('')
        # imgPrinter.printImageBuffer(sharedData.TrainingDataSet.meanVectorByLabel.get(i))

        print(colors.bcolors.HEADER + ' calculating CoV Matrix for Dataset_Label: ' + colors.bcolors.OKBLUE + str(i))
        sharedData.TrainingDataSet.covarianceMatrixByLabel[i] = mh.calculateCovarianceByIndexes(
            meanVector=sharedData.TrainingDataSet.meanVectorByLabel[i], indexes=values, dynamicSize=_dynamicSize)

        covarianceMatrixByLabel = np.array(sharedData.TrainingDataSet.covarianceMatrixByLabel[i])
        mh.calculatePreDiscriminantParameters(index=i)

    # calculating g(x) for every set sets and
    print('Validating Tests...')

    for i in tqdm (range(len(sharedData.TestSetData.labelsDataSet))):
        calculateDiscriminant(sharedData.TestSetData.imagesDataSet[i], sharedData.TestSetData.labelsDataSet[i])

    print('------------')
    i =0
    for key in sorted(sharedData.Results.resultByIndex):
        print('C is: ' + str(key) + ' predicted: |' + str(sharedData.Results.resultByIndex[key][:]) + ' | ==> Accuracy: ' +
              str(sharedData.Results.resultByIndex[key][i]*100 / sum(sharedData.Results.resultByIndex[key][:]))+ ' %')
        i += 1

    print('number of samples: ' + str(sharedData.Results.number_of_all_samples))
    print('number of correct: ' + str(sharedData.Results.number_of_corrects))


def __init__():
    loadData()
    startBeysian(_dynamicSize=False)

