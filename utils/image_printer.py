import numpy as np
import matplotlib.pyplot as plt

import configuration.config as config
import configuration.sharedData as sharedData


def printImageLabel(labelData, imgData, index):
    print(labelData[index])
    data = np.array(imgData[index])
    data = data.reshape((config.SAMPLE_IMAGE_SIZE, config.SAMPLE_IMAGE_SIZE))

    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()


def printImageLabel(index):
    if not sharedData.TrainingDataSet.labelsDataSet or not sharedData.TrainingDataSet.imagesDataSet :
        print("TrainingDataSet is empty")
        return

    print(sharedData.TrainingDataSet.labelsDataSet[index])
    data = np.array(sharedData.TrainingDataSet.imagesDataSet[index])
    data = data.reshape((config.SAMPLE_IMAGE_SIZE, config.SAMPLE_IMAGE_SIZE))

    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()