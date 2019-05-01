import os
import struct
import gzip
import sys
import configuration.config as config
import utils.base_utils.colors as colors
import configuration.sharedData as sharedData
import utils.image_printer as imagePrinter
import numpy as np


def loadTrainingSets(testIndex, fast, sampleSize, dt_type):

    fileDir = os.path.dirname(os.path.realpath('__file__'))

    imgFile = None
    labelFile = None

    if dt_type == config.dataset_type.mnist_hw:
        labelFile = open(os.path.join(fileDir, config.MNIST_HW_LABEL_TRAINING_SET_PATH), "rb")
        imgFile = gzip.open(os.path.join(fileDir, config.MNIST_HW_IMAGE_TRAINING_SET_PATH), 'r')

    elif dt_type == config.dataset_type.mnist_fashion:
        labelFile = open(os.path.join(fileDir, config.MNIST_FASHION_LABEL_TRAINING_SET_PATH), "rb")
        imgFile = gzip.open(os.path.join(fileDir, config.MNIST_FASHION_IMAGE_TRAINING_SET_PATH), 'r')

    if imgFile is None or labelFile is None:
        return None

    imageSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE
    imageFlatSize = imageSize * imageSize
    labelsDataSet = []

    magicNumber, numberOfSamples = struct.unpack(">II", labelFile.read(8))

    imagesDataSet = []
    imagesDataSet_3d = np.zeros((60000, config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE))

    imgFile.read(16)

    if fast:
        numberOfSamples = sampleSize
        print("FAST_MODE")

    print("")
    print(colors.bcolors.UNDERLINE + "number of samples are : ", numberOfSamples)
    print()

    # print("magic number: ", magicNumber)

    print(colors.bcolors.HEADER + ' Loading data Started...')

    try:
        for i in range(numberOfSamples):

            # importing label data
            label = struct.unpack('B', labelFile.read(1))[0]
            labelsDataSet.append(label)
            sharedData.TrainingDataSet.labelIndexes[label].append(i)

            # importing image data
            imgDataArray = []
            for k in range(imageFlatSize):
                pixelData = struct.unpack('B', imgFile.read(1))[0]
                imgDataArray.append(pixelData/255)

            msg = '\033[92m' + " sample number " + str(i + 1) + \
                  " created." + '\033[94m' + "remains " + str(numberOfSamples - i - 1)

            sys.stdout.write('\r' + msg)

            imagesDataSet.append(imgDataArray)
            imagesDataSet_3d[i] = (np.array(imgDataArray).reshape(-1, 28))

    finally:
        imgFile.close()
        labelFile.close()

    print('  ')
    print('')
    print(colors.bcolors.HEADER + ' Loading data completed....')

    sharedData.TrainingDataSet.labelsDataSet = labelsDataSet
    sharedData.TrainingDataSet.imagesDataSet = imagesDataSet
    sharedData.TrainingDataSet.img_ds_3d = imagesDataSet_3d

    if testIndex > -1:
        imagePrinter.printImageLabel(testIndex)
        # printImageLabel(labelsDataSet, imagesDataSet, testIndex)


def loadTestSets(dt_type):

    fileDir = os.path.dirname(os.path.realpath('__file__'))

    imgFile = None
    labelFile = None

    if dt_type == config.dataset_type.mnist_hw:
        labelFile = open(os.path.join(fileDir, config.MNIST_HW_LABEL_TEST_SET_PATH), "rb")
        imgFile = gzip.open(os.path.join(fileDir, config.MNIST_HW_IMAGE_TEST_SET_PATH), 'r')

    elif dt_type == config.dataset_type.mnist_fashion:
        labelFile = open(os.path.join(fileDir, config.MNIST_FASHION_LABEL_TEST_SET_PATH), "rb")
        imgFile = gzip.open(os.path.join(fileDir, config.MNIST_FASHION_IMAGE_TEST_SET_PATH), 'r')

    if imgFile is None or labelFile is None:
        return None

    imageSize = config.MNIST_DS_SAMPLE_IMAGE_SIZE
    imageFlatSize = imageSize * imageSize
    labelsDataSet = []

    magicNumber, numberOfSamples = struct.unpack(">II", labelFile.read(8))

    imagesDataSet = []
    imagesDataSet_3d = np.zeros((10000, config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE))

    imgFile.read(16)

    print("")
    print(colors.bcolors.UNDERLINE + "number of tests are : ", numberOfSamples)
    print()

    # print("magic number: ", magicNumber)

    print(colors.bcolors.HEADER + ' Loading test data Started...')

    try:
        for i in range(numberOfSamples):

            # importing label data
            label = struct.unpack('B', labelFile.read(1))[0]
            labelsDataSet.append(label)
            # sharedData.TrainingDataSet.labelIndexes[label].append(i)

            # importing image data
            imgDataArray = []
            for k in range(imageFlatSize):
                pixelData = struct.unpack('B', imgFile.read(1))[0]
                imgDataArray.append(pixelData/255)

            msg = '\033[92m' + " sample number " + str(i + 1) + \
                  " created." + '\033[94m' + "remains " + str(numberOfSamples - i - 1)

            sys.stdout.write('\r' + msg)

            imagesDataSet.append(imgDataArray)
            imagesDataSet_3d[i] = (np.array(imgDataArray).reshape(-1, 28))

    finally:
        imgFile.close()
        labelFile.close()

    print('  ')
    print('')
    print(colors.bcolors.HEADER + ' Loading test data completed....')

    sharedData.TestSetData.labelsDataSet = labelsDataSet
    sharedData.TestSetData.imagesDataSet = imagesDataSet
    sharedData.TestSetData.img_ds_3d = imagesDataSet_3d




