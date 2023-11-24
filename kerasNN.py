import keras
import tensorflow as tf

from constants import *


def bulidMnistMLP(inputShape):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=inputShape),
        keras.layers.Dense(
            MnistMLP.NEURON_COUNT, activation='relu',
            kernel_initializer='he_normal', bias_initializer='he_normal'
        ),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics='sparse_categorical_accuracy')
    return model


def buildMnistCNN(inputShape):
    model = keras.models.Sequential([
        keras.layers.Conv2D(
            MnistCNN.FILTER, MnistCNN.KERNEL, MnistCNN.STRIDES,
            input_shape=(*inputShape, 1), padding='valid', activation='relu',
            kernel_initializer='he_normal', bias_initializer='he_normal'
        ),
        keras.layers.MaxPooling2D(
            MnistCNN.POOL_SIZE, MnistCNN.POOL_STRIDE, padding='valid'),
        keras.layers.Flatten(),
        keras.layers.Dense(
            MnistCNN.DENSE_COUNT, activation='relu',
            kernel_initializer='he_normal', bias_initializer='he_normal'
        ),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics='sparse_categorical_accuracy')
    return model


def buildCifar10CNN(inputShape):
    model = keras.Sequential([
        keras.layers.Conv2D(
            Cifar10CNN.FILTER, Cifar10CNN.KERNEL, Cifar10CNN.STRIDES,
            input_shape=inputShape, padding='valid', activation='relu',
            kernel_initializer='he_normal', bias_initializer='he_normal'
        ),
        keras.layers.AveragePooling2D(
            Cifar10CNN.POOL_SIZE, Cifar10CNN.POOL_STRIDE),
        keras.layers.Flatten(),
        keras.layers.Dense(
            Cifar10CNN.DENSE_COUNT, activation='relu',
            kernel_initializer='he_normal', bias_initializer='he_normal'
        ),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics="sparse_categorical_accuracy")
    return model


def run(modelName, data):
    if modelName == 'mnistMLP':
        model = bulidMnistMLP(data['xTrain'].shape[1:])
    elif modelName == 'mnistCNN':
        model = buildMnistCNN(data['xTrain'].shape[1:])
    elif modelName == 'cifar10CNN':
        model = buildCifar10CNN(data['xTrain'].shape[1:])
    if 0 < VERBOSE < 3:
        model.summary()

    print('\nTraining '+modelName)
    with tf.device('/device:GPU:0'):
        model.fit(
            data['xTrain'], data['yTrain'], epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(data['xVal'], data['yVal']),
            callbacks=keras.callbacks.EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=PATIENCE, restore_best_weights=True),
            verbose=VERBOSE
        )

    print('\nEvaluating '+modelName)
    res = model.evaluate(data['xTest'], data['yTest'], verbose=VERBOSE)
    print(model.metrics_names[0], '%.4f' % res[0])
    print(model.metrics_names[1], '%.4f' % res[1], '\n')
    return model


def runMnistMLP(data):
    run('mnistMLP', data)


def runMnistCNN(data):
    run('mnistCNN', data)


def runCifar10CNN(data):
    run('cifar10CNN', data)
