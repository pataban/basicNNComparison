from keras.datasets import mnist

from constants import *
from dataCreation import *
from Network import Network



if __name__=='__main__':
    
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX.shape=(60000,784,1)
    trainX=trainX/255.0
    testX.shape=(10000,784,1)
    testX=testX/255.0
    print("\nData load completed")


    network=Network()
    network.fit(trainX[0:1000],trainY[0:1000],testX,testY)
    print("Learn completed")


    print(network.test(testX[0:TEST_SIZE],testY[0:TEST_SIZE]))
    
    

