import os
import struct
import gzip
import sys

import configuration.config as config
import utils.base_utils.colors as colors
import configuration.sharedData as sharedData
import utils.image_printer as imagePrinter


def loadTrainingSets(testIndex, fast):

    fileDir = os.path.dirname(os.path.realpath('__file__'))

    labelFile = open(os.path.join(fileDir, config.LABEL_TRAINING_SET_PATH), "rb")
    imgFile = gzip.open(os.path.join(fileDir, config.IMAGE_TRAINING_SET_PATH), 'r')

    imageSize = config.SAMPLE_IMAGE_SIZE
    imageFlatSize = imageSize * imageSize
    labelsDataSet = []

    magicNumber, numberOfSamples = struct.unpack(">II", labelFile.read(8))

    imagesDataSet = []
    imgFile.read(16)

    if fast:
        numberOfSamples = 100
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
                imgDataArray.append(pixelData)

            msg = '\033[92m' + " sample number " + str(i + 1) + \
                  " created." + '\033[94m' + "remains " + str(numberOfSamples - i - 1)
            sys.stdout.write('\r' + msg)

            imagesDataSet.append(imgDataArray)
    finally:
        imgFile.close()
        labelFile.close()

    print('  ')
    print('')
    print(colors.bcolors.HEADER + ' Loading data completed....')

    sharedData.TrainingDataSet.labelsDataSet = labelsDataSet
    sharedData.TrainingDataSet.imagesDataSet = imagesDataSet

    if testIndex > -1:
        imagePrinter.printImageLabel(testIndex)
        # printImageLabel(labelsDataSet, imagesDataSet, testIndex)


