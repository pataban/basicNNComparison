import random
import numpy as np
from math import exp

from constants import *


class Layer():

    def __init__(self, inputSize, neuronCount, activateFun, weightInit, biasInit):
        self.w = Layer.WEIGHT_INITIALIZERS[weightInit](neuronCount, inputSize)
        self.b = Layer.BIAS_INITIALIZERS[biasInit](neuronCount)
        self.aFun, self.aFunDer = Layer.ACTIVATION_FUNCTIONS[activateFun]

    def calc(self, xData):
        z = np.matmul(self.w, xData) + self.b
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
    def sigmoidalDerivative(z):
        pass

    @staticmethod
    def tanHDerivative(z):
        return np.vectorize(lambda ai: 1-ai**2)(z)

    @staticmethod
    def reLUDerivative(z):
        return np.vectorize(lambda ai: 0.0 if ai <= 0.0 else 1.0)(z)

    @staticmethod
    def softMaxDerivative(z, yData):
        return z-yData

    ACTIVATION_FUNCTIONS = {
        'sigmoidal': (sigmoidal, sigmoidalDerivative),
        'tanH': (tanH, tanHDerivative),
        'reLU': (reLU, reLUDerivative),
        'softMax': (softMax, softMaxDerivative),
    }

    @staticmethod
    def makeWeightZero(neurons, inputs):
        return np.full((neurons, inputs), fill_value=0.0)

    @staticmethod
    def makeWeightRandom(neurons, inputs):
        w = np.empty((neurons, inputs), float)
        for i in range(neurons):
            for j in range(inputs):
                w[i, j] = random.random()*ManualMLP.WEIGHTS_RANDOM_MAX * \
                    random.choice([-1, 1])
        return w

    WEIGHT_INITIALIZERS = {
        'zero': makeWeightZero,
        'random': makeWeightRandom
    }

    @staticmethod
    def makeBiasZero(neurons):
        return np.full((neurons, 1), fill_value=0.0)

    @staticmethod
    def makeBiasRandom(neurons):
        bias = np.empty((neurons, 1), float)
        for i in range(neurons):
            bias[i] = random.random()*ManualMLP.BIAS_RANDOM_MAX * \
                (random.choice([-1, 1])
                 if ManualMLP.BIAS_RANDOM_NEGATIVE else 1)
        return bias

    BIAS_INITIALIZERS = {
        'zero': makeBiasZero,
        'random': makeBiasRandom
    }
