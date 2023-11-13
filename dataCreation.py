from constants import *
import numpy as np
import random

def makeBias(n):
    bias=np.empty((n,1),float)
    for i in range(n):
        bias[i]=random.random()*BIAS_RANDOM_MAX *(random.choice([-1,1]) if BIAS_RANDOM_NEGATIVE else 1)
    return bias

def makeWeights(n,k):
    w=np.empty((n,k),float)
    for i in range(n):
        for j in range(k):
            w[i,j]=random.random()*WEIGHTS_RANDOM_MAX*random.choice([-1,1])
    return w

def makeBiasZero(n):
    return np.full((n,1),fill_value=0.0)

def makeWeightsZero(n,k):
    return np.full((n,k),fill_value=0.0)
