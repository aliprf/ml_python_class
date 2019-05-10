#!/usr/bin/env python
import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
from sklearn.svm import SVC
from sklearn import metrics
import configuration.sharedData as shd
import configuration.config as config
import threading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def startLDA(data):

    # svmClassifier = SVC(probability=False, kernel=kernel, C=c_parameter, gamma=.0073)

    print("Start LDA")

    trainingSize = len(data['train']['img'])
    svmClassifier = LDA()
    svmClassifier.fit(data['train']['img'][:trainingSize], data['train']['lbl'][:trainingSize])

    analyze(svmClassifier, data)


def analyze(clf, data):

    predicted = clf.predict(data['test']['img'])

    file = open('LDA_result.txt', 'w')

    file.write("\nConfusion matrix:\n %s" % metrics.confusion_matrix(data['test']['lbl'], predicted))

    file.write("\nAccuracy: %s" % metrics.accuracy_score(data['test']['lbl'], predicted))

    file.close()


def processData():

    dataLoader.loadTrainingSets(testIndex=-1, fast=False, sampleSize=150, dt_type=config.dataset_type.mnist_hw)
    dataLoader.loadTestSets(dt_type=config.dataset_type.mnist_hw)

    data = {'train': {'img': shd.TrainingDataSet.imagesDataSet,
                      'lbl': shd.TrainingDataSet.labelsDataSet},
            'test': {'img': shd.TestSetData.imagesDataSet,
                     'lbl': shd.TestSetData.labelsDataSet}}

    shd.TrainingDataSet.imagesDataSet = None
    shd.TrainingDataSet.labelsDataSet = None
    shd.TestSetData.imagesDataSet = None
    shd.TestSetData.labelsDataSet = None

    return data


if __name__ == '__main__':
    data = processData()
    startLDA(data)
