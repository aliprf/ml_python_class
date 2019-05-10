import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import math
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.auto
tqdm.tqdm= tqdm.auto.tqdm

import utils.input_data_formatter.mnist.mnistDataLoader as dataLoader
import configuration.sharedData as sharedData
import utils.mathHelper as mh
import utils.image_printer as imgPrinter
import utils.base_utils.colors as colors
import numpy as np
from tqdm import tqdm
import configuration.config as config


# imgPrinter.print_images(20, 30)
# nnn = np.array(sharedData.TrainingDataSet.imagesDataSet[0])
# input_data = np.array(sharedData.TrainingDataSet.imagesDataSet[0]).reshape(-1, 28)
input_data = sharedData.TrainingDataSet.img_ds_3d




def load_data():
    dataLoader.loadTrainingSets(testIndex=-1, fast=True, sampleSize=100, dt_type=config.dataset_type.mnist_fashion)
    dataLoader.loadTestSets(dt_type=config.dataset_type.mnist_fashion)


def pre_process_data():


def learn():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(config.MNIST_DS_SAMPLE_IMAGE_SIZE, config.MNIST_DS_SAMPLE_IMAGE_SIZE, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(input_data, epochs=5)

    model.predict(sharedData.TestSetData.imagesDataSet[1])
    print(sharedData.TestSetData.labelsDataSet[1])



if __name__ == '__main__':
    load_data()
    pre_process_data()