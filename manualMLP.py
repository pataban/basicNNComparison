from Network import Network
from constants import *
from dataCreation import *


def runManualMLP(data):
    orginalShape = data['xTrain'].shape[1:]
    for k in ['xTrain', 'xVal', 'xTest']:
        data[k].shape = (data[k].shape[0], data[k].shape[1]
                         * data[k].shape[2], 1)

    print('\nTraining manualMLP')
    model = Network()
    model.fit(data['xTrain'], data['yTrain'], data['xVal'], data['yVal'])

    print('\nEvaluating manualMLP')
    print(model.evaluate(data['xTest'], data['yTest']), '\n')

    for k in ['xTrain', 'xVal', 'xTest']:
        data[k].shape = [data[k].shape[0]]+list(orginalShape)
