import random
import numpy as np
import tensorflow as tf
from math import exp

from constants import *


class Layer():

    def __init__(self, inputSize, neuronCount, activateFun, weightInit, biasInit):
        self.w = Layer.WEIGHT_INITIALIZERS[weightInit](neuronCount, inputSize)
        self.b = Layer.BIAS_INITIALIZERS[biasInit](neuronCount)
        self.aFun, self.aFunDer = Layer.ACTIVATION_FUNCTIONS[activateFun]

    def calc(self, xData):
        z = tf.matmul(self.w, xData) + self.b
        a = self.aFun(z)
        return a

    @staticmethod
    def sigmoidal(z):
        return tf.constant(np.vectorize(lambda zi: 1/(1+exp(-zi)))(z))

    @staticmethod
    def tanH(z):
        return tf.constant(np.vectorize(lambda zi: 2/(1+exp(-2*zi))-1)(z))

    @staticmethod
    def reLU(z):
        # fail retracing
        """
        @tf.function
        def reLuOne(zi):
            return tf.cond(zi < 0.0, lambda: tf.constant(0.0, dtype=tf.float64, shape=()),
                           lambda zi=zi: zi)
        zOrgShape = z.shape
        z = tf.reshape(z, (zOrgShape[0]*zOrgShape[1]))
        a = tf.reshape(tf.vectorized_map(Layer.reLuOne, z), zOrgShape)
        return a
        """
        return tf.constant(np.vectorize(lambda zi: 0.0 if zi < 0.0 else zi)(z))

    @staticmethod
    def softMax(z):
        z = tf.constant(np.vectorize(exp)(z))
        maxSum = tf.reduce_sum(z, axis=1)
        z = tf.transpose(tf.transpose(z, perm=(1, 0, 2)) /
                         maxSum, perm=(1, 0, 2))
        return z

    @staticmethod
    def sigmoidalDerivative(z):
        pass

    @staticmethod
    def tanHDerivative(z):
        return tf.constant(np.vectorize(lambda ai: 1-ai**2)(z))

    @staticmethod
    def reLUDerivative(z):
        return tf.constant(np.vectorize(lambda ai: 0.0 if ai <= 0.0 else 1.0)(z))

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
        return tf.fill((neurons, inputs), value=0.0)

    @staticmethod
    def makeWeightRandom(neurons, inputs):
        w = np.empty((neurons, inputs), float)
        for i in range(neurons):
            for j in range(inputs):
                w[i, j] = random.random()*ManualMLP.WEIGHTS_RANDOM_MAX * \
                    random.choice([-1, 1])
        return tf.constant(w)

    WEIGHT_INITIALIZERS = {
        'zero': makeWeightZero,
        'random': makeWeightRandom
    }

    @staticmethod
    def makeBiasZero(neurons):
        return tf.fill((neurons, 1), value=0.0)

    @staticmethod
    def makeBiasRandom(neurons):
        bias = np.empty((neurons, 1), float)
        for i in range(neurons):
            bias[i] = random.random()*ManualMLP.BIAS_RANDOM_MAX * \
                (random.choice([-1, 1])
                 if ManualMLP.BIAS_RANDOM_NEGATIVE else 1)
        return tf.constant(bias)

    BIAS_INITIALIZERS = {
        'zero': makeBiasZero,
        'random': makeBiasRandom
    }
