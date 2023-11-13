from constants import *
from dataCreation import *
from math import exp


class Layer():
    def __init__(self,neuronInputCount,neuronCount,activateFun,activateFunDer=None):
        self.w=makeWeights(neuronCount,neuronInputCount)
        self.b=makeBias(neuronCount)
        self.aFun=activateFun
        self.activateFunDer=activateFunDer
    
    def calc(self,x):
        z=np.matmul(self.w,x) + self.b
        a=None
        a=self.aFun(z)
        return a
    

    def sigmoidal(z):
        return np.vectorize(lambda zi: 1/(1+exp(-zi)))(z)

    def tanH(z):
        return np.vectorize(lambda zi: 2/(1+exp(-2*zi))-1)(z)

    def reLU(z):
        return np.vectorize(lambda zi: 0.0 if zi < 0.0 else zi)(z)

    def softMax(z):
        z=np.vectorize(lambda zi: exp(zi))(z)
        maxSum=z.sum(axis=1)
        for i in range(z.shape[0]):
            z[i]/=maxSum[i]
        return z

    
    def sigmoidalDerivative(a):
        pass

    def tanHDerivative(a):
        return np.vectorize(lambda ai: 1-ai**2)(a)

    def reLUDerivative(a):
        return np.vectorize(lambda ai: 0.0 if ai <= 0.0 else 1.0)(a)

