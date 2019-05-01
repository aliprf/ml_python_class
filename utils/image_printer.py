import numpy as np
import matplotlib.pyplot as plt

import configuration.config as config
import configuration.sharedData as sharedData


def printImageBuffer(imgData, x, y):
    data = np.array(imgData)
    if x is not None:
        data = data.reshape((x, y))
    else:
        data = data.reshape((config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE))

    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()


def printImageLabel(labelData, imgData, index):
    print(labelData[index])
    data = np.array(imgData[index])
    data = data.reshape((config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE))

    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()


def printImageLabel(index):
    if not sharedData.TrainingDataSet.labelsDataSet or not sharedData.TrainingDataSet.imagesDataSet:
        print("TrainingDataSet is empty")
        return

    print(sharedData.TrainingDataSet.labelsDataSet[index])
    data = np.array(sharedData.TrainingDataSet.imagesDataSet[index])
    data = data.reshape((config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE))

    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()


def print_images(index_from, index_to):

    plt.figure(figsize=(10, 10))
    i =0

    for index in range(index_from, index_to):

        image = np.array(sharedData.TrainingDataSet.imagesDataSet[index])
        image = image.reshape((config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE))

        label = sharedData.TrainingDataSet.labelsDataSet[index]

        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(config.mnist_fashion_class_names[label])
        i += 1
    plt.show()


def printEigenValues(datasetLabel):

    lenEigenValues = len(sharedData.TrainingDataSet.eigenValueByLabel[datasetLabel])
    sumTotal = 0
    sumByValue = 0
    wieghtArray = []
    for i in range(lenEigenValues):
        sumTotal += sharedData.TrainingDataSet.eigenValueByLabel[datasetLabel][i]

    for i in range(lenEigenValues):
        itemValue = sharedData.TrainingDataSet.eigenValueByLabel[datasetLabel][i]
        eigPercent = itemValue * 100 / sumTotal
        if eigPercent > 0.1:
            wieghtArray.append(eigPercent)

    plt.bar(np.arange(len(wieghtArray)), wieghtArray)
    # plt.step(range(4), cum_var_exp, where='mid',
    #          label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.show()