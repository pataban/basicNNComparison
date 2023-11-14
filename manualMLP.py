import os
# or any {'0' info, '1' warning, '2' error, '3' fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from Network import Network
from constants import *
from dataCreation import *


if __name__ == '__main__':

    print('\nLoading data')
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain.shape = (xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2], 1)
    xTrain = xTrain/255.0

    xTest.shape = (xTest.shape[0], xTest.shape[1] * xTest.shape[2], 1)
    xTest = xTest/255.0

    xTrain, xVal, yTrain, yVal = train_test_split(
        xTrain, yTrain, test_size=VALIDATION_SIZE)

    print('\nTraining\n')
    model = Network()
    model.fit(xTrain[0:TRAIN_SIZE], yTrain[0:TRAIN_SIZE], xVal, yVal)

    print('\nEvaluating')
    print(model.evaluate(xTest[0:TEST_SIZE], yTest[0:TEST_SIZE]))
