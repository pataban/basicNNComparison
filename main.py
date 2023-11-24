import os
# or any {'0' info, '1' warning, '2' error, '3' fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.datasets import mnist
import tensorflow as tf

from kerasNN import runMnistMLP, runMnistCNN, runCifar10CNN
from manualMLP import runManualMLP
from constants import *


def printShow(x, y):
    print(x)
    print(Cifar10CNN.LABELS[y[0]])
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    print('\nAvailable GPU:', tf.config.list_physical_devices('GPU'))

    print('\nLoading data')
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = xTrain/255.0
    xTest = xTest/255.0
    xTrain, xVal, yTrain, yVal = train_test_split(
        xTrain, yTrain, test_size=VAL_SIZE)
    data = {
        'mnist': {
            'xTrain': xTrain[0:TRAIN_SIZE], 'yTrain': yTrain[0:TRAIN_SIZE],
            'xVal': xVal, 'yVal': yVal,
            'xTest': xTest[0:TEST_SIZE], 'yTest': yTest[0:TEST_SIZE]
        }
    }

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    # printShow(1)
    xTrain = xTrain/255.0
    xTest = xTest/255.0
    xTrain, xVal, yTrain, yVal = train_test_split(
        xTrain, yTrain, test_size=VAL_SIZE)
    data['cifar10'] = {
        'xTrain': xTrain[0:TRAIN_SIZE], 'yTrain': yTrain[0:TRAIN_SIZE],
        'xVal': xVal, 'yVal': yVal,
        'xTest': xTest[0:TEST_SIZE], 'yTest': yTest[0:TEST_SIZE]
    }

    runManualMLP(data['mnist'])
    runMnistMLP(data['mnist'])
    runMnistCNN(data['mnist'])
    runCifar10CNN(data['cifar10'])
