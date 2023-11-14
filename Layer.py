import numpy as np
from math import exp
from constants import *
from dataCreation import *


class Layer():

    def __init__(self, inputSize, neuronCount, activateFun):
        self.w = makeWeights(neuronCount, inputSize)
        self.b = makeBias(neuronCount)
        self.aFun, self.aFunDer = Layer.activationFunctions[activateFun]

    def calc(self, x):
        z = np.matmul(self.w, x) + self.b
        a = None
        a = self.aFun(z)
        return a

    @staticmethod
    def sigmoidal(z):
        return np.vectorize(lambda zi: 1/(1+exp(-zi)))(z)

    @staticmethod
    def tanH(z):
        return np.vectorize(lambda zi: 2/(1+exp(-2*zi))-1)(z)

    @staticmethod
    def reLU(z):
        return np.vectorize(lambda zi: 0.0 if zi < 0.0 else zi)(z)

    @staticmethod
    def softMax(z):
        z = np.vectorize(exp)(z)
        maxSum = z.sum(axis=1)
        for i in range(z.shape[0]):
            z[i] /= maxSum[i]
        return z

    @staticmethod
    def sigmoidalDerivative(a):
        pass

    @staticmethod
    def tanHDerivative(a):
        return np.vectorize(lambda ai: 1-ai**2)(a)

    @staticmethod
    def reLUDerivative(a):
        return np.vectorize(lambda ai: 0.0 if ai <= 0.0 else 1.0)(a)

    activationFunctions = {
        'sigmoidal': (sigmoidal, sigmoidalDerivative),
        'tanH': (tanH, tanHDerivative),
        'reLU': (reLU, reLUDerivative),
        'softMax': (softMax, None),
    }
