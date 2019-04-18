import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
import configuration.sharedData as sharedData
import utils.mathHelper as mh
import utils.image_printer as imgPrinter
import utils.base_utils.colors as colors
import numpy as np


#
def calculateDiscriminant(inputVector, label):  # the input is array, so we consider it as a transpose of the vector
    print("label")
    print(label)

    inputVector = np.array(inputVector)[np.newaxis]
    tr_inputVector = inputVector.T

    discArr = []
    for i, values in sharedData.TrainingDataSet.labelIndexes.items():
        gx1= np.dot(np.dot(inputVector, sharedData.TrainingDataSet.W_matrix_ByLabel[i]), tr_inputVector)
        tmp= sharedData.TrainingDataSet.w_matrix_ByLabel[i]
        gx2= np.dot(tmp,tr_inputVector)
        gx3= sharedData.TrainingDataSet.w_zero_ByLabel[i]

        discArr.append(gx1[0][0]+gx2[0][0])

    print("label")
    print(label)
    print("discArr")
    print(discArr)


dataLoader.loadTrainingSets(testIndex=-1, fast=True, sampleSize=100)
dataLoader.loadTestSets()

# calculating mean and covariance for each dataset
for i, values in sharedData.TrainingDataSet.labelIndexes.items():
    print(colors.bcolors.HEADER + ' calculating Mean Vector for Dataset_Label: ' + colors.bcolors.OKBLUE + str(i))
    sharedData.TrainingDataSet.meanVectorByLabel[i] = mh.calculateMeanByIndexes(indexes=values, needTest=False)

    print('')
    # imgPrinter.printImageBuffer(sharedData.TrainingDataSet.meanVectorByLabel.get(i))

    print(colors.bcolors.HEADER + ' calculating CoV Matrix for Dataset_Label: ' + colors.bcolors.OKBLUE + str(i))
    sharedData.TrainingDataSet.covarianceMatrixByLabel[i] = mh.calculateCovarianceByIndexes(
        meanVector=sharedData.TrainingDataSet.meanVectorByLabel[i], indexes=values)

    mh.calculatePreDiscriminantParameters(index=i)


# calculating g(x) for every set sets and
for i in range(len(sharedData.TestSetData.labelsDataSet)):
    calculateDiscriminant(sharedData.TestSetData.imagesDataSet[i], sharedData.TestSetData.labelsDataSet[i])
