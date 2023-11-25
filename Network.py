import numpy as np

from constants import *


class Network():
    def __init__(self, layers):
        self.layers = layers

    def fit(self, xTrain, yTrain, xVal, yVal):
        yTrain = Network.toOneHotNotation(yTrain)
        epoch = 1
        successRate = 0.0
        bestSuccessRate = -1.0
        while (bestSuccessRate + ManualMLP.SUCCESS_EARLY_STOP <
               successRate < ManualMLP.SUCCESS_MIN):
            print("epoch: "+f"{epoch} ".rjust(3), end='')
            for i in range(0, xTrain.shape[0], ManualMLP.BATCH_SIZE):
                if int(i/ManualMLP.BATCH_SIZE) % ManualMLP.PROGRESS_CHECK == 0:
                    print(".", end='')
                self.fitBatch(xTrain[i:i+ManualMLP.BATCH_SIZE],
                              yTrain[i:i+ManualMLP.BATCH_SIZE])

            if successRate > bestSuccessRate:
                bestSuccessRate = successRate
            successRate = self.evaluate(xVal, yVal)
            print(" -> %.4lf" % (successRate))
            epoch += 1

    def fitBatch(self, xBatch, yBatch):
        for layer in self.layers:  # expand to batch size
            layer.w = np.stack([layer.w] * xBatch.shape[0], axis=0)
            layer.b = np.stack([layer.b] * xBatch.shape[0], axis=0)

        a = [xBatch]
        z = []
        for layer in self.layers:
            z.append(np.matmul(layer.w, a[-1]) + layer.b)
            a.append(layer.aFun(z[-1]))

        # empty list of length == len(layers)
        grads = list(range(0, len(self.layers)))
        grads[-1] = self.layers[-1].aFunDer(z[-1], yBatch)
        # print(grads[-1])                                              #10x1
        for i in reversed(range(0, len(self.layers)-1)):
            # print(self.layers[i+1].w.transpose((0, 2, 1).shape)       #150x10
            # print(grads[i+1].shape)                                   #10x1
            # print(self.layers[i].aFunDer(z[i]).shape)                 #150x1
            grads[i] = np.matmul(
                self.layers[i+1].w.transpose((0, 2, 1)), grads[i+1])  # 150x1
            grads[i] *= self.layers[i].aFunDer(z[i])
            # print(grads[i])                                           #150x1

        # shrink to normal size and update weights
        for layer, grad, ai in zip(self.layers, grads, a):
            aGradSum = np.matmul(grad, ai.transpose((0, 2, 1))).sum(axis=0)
            layer.w = layer.w[0] - ManualMLP.LEARNING_SPEED * aGradSum
            layer.b = layer.b[0] - ManualMLP.LEARNING_SPEED * grad.sum(axis=0)

    def classify(self, xData):
        for layer in self.layers:
            xData = layer.calc(xData)
        return xData.argmax(axis=1).flatten()

    def evaluate(self, xData, yData):
        classification = self.classify(xData)

        res = (classification-yData).astype(bool).astype(int)

        successRate = (xData.shape[0]-res.sum())/xData.shape[0]
        return successRate

    @staticmethod
    def toOneHotNotation(yData):
        newY = np.full((yData.shape[0], 10, 1), fill_value=0)
        for i, y in enumerate(yData):
            newY[i, y] = 1
        return newY
