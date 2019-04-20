import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
import configuration.sharedData as sharedData
import utils.mathHelper as mh
import utils.image_printer as imgPrinter
import utils.base_utils.colors as colors
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import threading


def calculateResults(testLabel, resultDic):
    sharedData.KNN_Results_1.number_of_all_samples += 1
    sharedData.KNN_Results_3.number_of_all_samples += 1
    sharedData.KNN_Results_5.number_of_all_samples += 1

    # for k1 --> get
    _key = sorted(resultDic)[0]
    predictedLabel = resultDic.get(_key)
    if predictedLabel == testLabel:
        sharedData.KNN_Results_1.number_of_corrects += 1

    rslArr = sharedData.KNN_Results_1.resultByIndex[testLabel]
    if rslArr is None or len(rslArr) == 0:
        rslArr = [0] * 10
        sharedData.KNN_Results_1.resultByIndex[testLabel] = rslArr
    else:
        rslArr[predictedLabel] += 1

    # for k3 --> get
    predictedLabelsArr_2 = []
    for i in range(3):
        _key = sorted(resultDic)[i]
        predictedLabelsArr_2.append(resultDic.get(_key))

    predictedLabel_2 = max(set(predictedLabelsArr_2), key=predictedLabelsArr_2.count)
    if predictedLabel_2 == testLabel:
        sharedData.KNN_Results_3.number_of_corrects += 1

    rslArr = sharedData.KNN_Results_3.resultByIndex[testLabel]
    if rslArr is None or len(rslArr) == 0:
        rslArr = [0] * 10
        sharedData.KNN_Results_3.resultByIndex[testLabel] = rslArr
    else:
        rslArr[predictedLabel_2] += 1

    # for k5 --> get
    predictedLabelsArr_3 = []
    for i in range(5):
        _key = sorted(resultDic)[i]
        predictedLabelsArr_3.append(resultDic.get(_key))

    predictedLabel_3 = max(set(predictedLabelsArr_3), key=predictedLabelsArr_3.count)
    if predictedLabel_3 == testLabel:
        sharedData.KNN_Results_5.number_of_corrects += 1

    rslArr = sharedData.KNN_Results_5.resultByIndex[testLabel]
    if rslArr is None or len(rslArr) == 0:
        rslArr = [0] * 10
        sharedData.KNN_Results_5.resultByIndex[testLabel] = rslArr
    else:
        rslArr[predictedLabel_3] += 1


def findNearestNeighbors(inputVector, _from, _to):
    distanceDic = defaultdict(list)
    # for i in range(len(sharedData.TrainingDataSet.imagesDataSet)):
    if _to > len(sharedData.TrainingDataSet.imagesDataSet):
        _to = len(sharedData.TrainingDataSet.imagesDataSet)

    for i in range(_from, _to):
        distance = mh.calculateDistance(inputVector, sharedData.TrainingDataSet.imagesDataSet[i])
        distanceDic[distance] = sharedData.TrainingDataSet.labelsDataSet[i]  # who care if distance for two or more is equal?

    globalDistanceDic.update(distanceDic)
    # return distanceDic


dataLoader.loadTrainingSets(testIndex=-1, fast=True, sampleSize=10000)
dataLoader.loadTestSets()

k_array = [1, 3, 5]
number_of_cpu = 8
globalDistanceDic = defaultdict(list)

for i in tqdm(range(len(sharedData.TestSetData.imagesDataSet))):
    testVector = sharedData.TestSetData.imagesDataSet[i]
    testLabel = sharedData.TestSetData.labelsDataSet[i]

    distanceDic = defaultdict(list)
    numberOfTasks = len(sharedData.TrainingDataSet.imagesDataSet)
    taskOffset = numberOfTasks // number_of_cpu

    threads = [None] * number_of_cpu

    for t in range(number_of_cpu):
        threads[t] = threading.Thread(target=findNearestNeighbors, args=(testVector, (t-1) * taskOffset, t * taskOffset))
        threads[t].start()

    for n in range(len(threads)):
        threads[n].join()

    # distanceDic = findNearestNeighbors(inputVector=testVector)
    distanceDic = globalDistanceDic
    calculateResults(testLabel=testLabel, resultDic=distanceDic)
    globalDistanceDic.clear()

print('----1NN--------')
i =0
for key in sorted(sharedData.KNN_Results_1.resultByIndex):
    print('C is: ' + str(key) + ' predicted: |' + str(sharedData.KNN_Results_1.resultByIndex[key][:]) + ' | ==> Accuracy: ' +
          str(sharedData.KNN_Results_1.resultByIndex[key][i]*100 / sum(sharedData.KNN_Results_1.resultByIndex[key][:]))+ ' %')
    i += 1
print('number of samples: ' + str(sharedData.KNN_Results_1.number_of_all_samples))
print('number of correct: ' + str(sharedData.KNN_Results_1.number_of_corrects))


print('----3NN--------')
i =0
for key in sorted(sharedData.KNN_Results_3.resultByIndex):
    print('C is: ' + str(key) + ' predicted: |' + str(sharedData.KNN_Results_3.resultByIndex[key][:]) + ' | ==> Accuracy: ' +
          str(sharedData.KNN_Results_3.resultByIndex[key][i]*100 / sum(sharedData.KNN_Results_3.resultByIndex[key][:]))+ ' %')
    i += 1
print('number of samples: ' + str(sharedData.KNN_Results_3.number_of_all_samples))
print('number of correct: ' + str(sharedData.KNN_Results_3.number_of_corrects))


print('----5NN--------')
i =0
for key in sorted(sharedData.KNN_Results_5.resultByIndex):
    print('C is: ' + str(key) + ' predicted: |' + str(sharedData.KNN_Results_5.resultByIndex[key][:]) + ' | ==> Accuracy: ' +
          str(sharedData.KNN_Results_5.resultByIndex[key][i]*100 / sum(sharedData.KNN_Results_5.resultByIndex[key][:]))+ ' %')
    i += 1

print('number of samples: ' + str(sharedData.KNN_Results_5.number_of_all_samples))
print('number of correct: ' + str(sharedData.KNN_Results_5.number_of_corrects))