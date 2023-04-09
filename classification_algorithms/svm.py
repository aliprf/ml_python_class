#!/usr/bin/env python
import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
from sklearn.svm import SVC
from sklearn import metrics
import configuration.sharedData as shd
import configuration.config as config
import threading


def startSVM(data, c_parameter, kernel):

    svmClassifier = SVC(probability=False, kernel=kernel, C=c_parameter, gamma=.0073)

    print("Start SVM with Parameters: " + '{Kernel: '+kernel+' , c: }' + str(c_parameter))

    trainingSize = len(data['train']['img'])

    svmClassifier.fit(data['train']['img'][:trainingSize], data['train']['lbl'][:trainingSize])

    analyze(svmClassifier, data, c_parameter, kernel)


def analyze(clf, data, c_parameter, kernel):

    predicted = clf.predict(data['test']['img'])

    file = open('svm_results/result_c_' + str(c_parameter)+'_kernel_'+kernel+'.txt', 'w')

    file.write("Parameters: " + '{Kernel: ' + kernel + ' , c: ' + str(c_parameter) + '} \n')

    file.write("\nConfusion matrix:\n %s" % metrics.confusion_matrix(data['test']['lbl'], predicted))

    file.write("\nAccuracy: %s" % metrics.accuracy_score(data['test']['lbl'], predicted))

    file.close()


def processData(readData):
    if readData :
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


def thread_svm_init(readData):
    # startSVM(c_parameter=0.0001, kernel="rbf")
    data = processData(readData=readData) #means load data from out side

    c_parameters = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    kernels = ["rbf", "linear"]

    number_of_cpu = 8
    threads = [None] * number_of_cpu
    #  rbf
    # i = 0
    # for t in range(number_of_cpu):
    #     threads[t] = threading.Thread(target=startSVM, args=(data, c_parameters[i], kernels[0]))
    #     threads[t].start()
    #     i += 1

    # linear
    i = 0
    for t in range(number_of_cpu):
        threads[t] = threading.Thread(target=startSVM, args=(data, c_parameters[i], kernels[1]))
        threads[t].start()
        i += 1

    # for n in range(len(threads)):
    #     threads[n].join()


if __name__ == '__main__':
    thread_svm_init(readData=True)
