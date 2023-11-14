import numpy as np
from Layer import Layer
from constants import *
from dataCreation import *


class Network():
    def __init__(self):
        self.layers = [
            Layer(NEURON_COUNT[0], NEURON_COUNT[1], 'reLU'),
            # Layer(NEURON_COUNT[1],NEURON_COUNT[2],'reLu'),
            # Layer(NEURON_COUNT[2],NEURON_COUNT[3],'reLu'),
            Layer(NEURON_COUNT[1], NEURON_COUNT[2], 'softMax')
        ]

    def fit(self, xTrain, yTrain, xVal, yVal):
        yTrain = Network.toOneHotNotation(yTrain)
        epoch = 1
        successRate = 0.0
        bestSuccessRate = -1.0
        while ((successRate < SUCCESS_MIN) and (successRate-bestSuccessRate > SUCCESS_EARLY_STOP)):
            print("epoch: "+f"{epoch} ".rjust(3), end='')
            for i in range(0, xTrain.shape[0], BATCH_SIZE):
                if int(i/BATCH_SIZE) % PROGRESS_CHECK == 0:
                    print(".", end='')
                self.fitBatch(xTrain[i:i+BATCH_SIZE], yTrain[i:i+BATCH_SIZE])

            if successRate > bestSuccessRate:
                bestSuccessRate = successRate
            successRate = self.evaluate(xVal[0:TEST_SIZE], yVal[0:TEST_SIZE])
            print(" -> %.4lf" % (successRate))
            epoch += 1

    def fitBatch(self, x, y):
        self.expandSize(x.shape[0])
        a = [x]
        z = []
        for layer in self.layers:
            z.append(np.matmul(layer.w, a[-1]) + layer.b)
            a.append(layer.aFun(z[-1]))

        # empty list of length == len(layers)
        grads = list(range(0, len(self.layers)))
        grads[-1] = -(y-z[-1])  # a[-2].T                         #10x1
        for i in reversed(range(0, len(self.layers)-1)):
            # print(self.layers[i+1].w.T.shape)                   #150x10
            # print(grads[i+1].shape)                             #10x1
            # print(self.layers[i].aFunDer(z[i]).shape )          #150x1
            # print(a[i].T.shape)                                 #1x300
            grads[i] = np.matmul(self.layers[i+1].w.transpose((0, 2, 1)),
                                 grads[i+1]) * self.layers[i].aFunDer(z[i])  # a[i].T

        # TODO move np.matmul(grad, ai.transpose((0, 2, 1))).sum(axis=0) here or this matmul to above
        for layer, grad, ai in zip(self.layers, grads, a):
            layer.w = layer.w[0] - LEARNING_SPEED * \
                np.matmul(grad, ai.transpose((0, 2, 1))).sum(axis=0)
            layer.b = layer.b[0] - LEARNING_SPEED * grad.sum(axis=0)

    def classify(self, x):
        for layer in self.layers:
            x = layer.calc(x)
        return x.argmax(axis=1).flatten()

    def evaluate(self, x, y):
        # print("y:\n",y)

        classes = self.classify(x)
        # print("classification:\n",classes.T)

        res = (classes-y).astype(bool).astype(int)
        # print("res:\n",res)

        successRate = (x.shape[0]-res.sum())/x.shape[0]
        # print("result",successRate)
        return successRate

    def expandSize(self, size=BATCH_SIZE):
        for layer in self.layers:
            # replicate shallow copy
            layer.w = np.stack([layer.w for _ in range(size)], axis=0)
            layer.b = np.stack([layer.b for _ in range(size)], axis=0)

    @staticmethod
    def toOneHotNotation(yData):
        newY = np.full((yData.shape[0], 10, 1), fill_value=0)
        for i, y in enumerate(yData):
            newY[i, y] = 1
        return newY
