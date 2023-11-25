from Layer import Layer
from Network import Network

from constants import *


def runManualMLP(data):
    orginalShape = data['xTrain'].shape[1:]
    for k in ['xTrain', 'xVal', 'xTest']:
        data[k].shape = (data[k].shape[0], data[k].shape[1]
                         * data[k].shape[2], 1)

    model = Network([
        Layer(data['xTrain'].shape[1], ManualMLP.NEURON_COUNT[0],
              'reLU', 'random', 'random'),
        # Layer(ManualMLP.NEURON_COUNT[0],ManualMLP.NEURON_COUNT[1],'reLu', 'random', 'random'),
        # Layer(ManualMLP.NEURON_COUNT[1],ManualMLP.NEURON_COUNT[2],'reLu', 'random', 'random'),
        Layer(ManualMLP.NEURON_COUNT[0], ManualMLP.NEURON_COUNT[1],
              'softMax', 'random', 'random')
    ])

    print('\nTraining manualMLP')
    model.fit(data['xTrain'], data['yTrain'], data['xVal'], data['yVal'])

    print('\nEvaluating manualMLP')
    print(model.evaluate(data['xTest'], data['yTest']), '\n')

    for k in ['xTrain', 'xVal', 'xTest']:
        data[k].shape = [data[k].shape[0]]+list(orginalShape)
