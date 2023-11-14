import os
# or any {'0' info, '1' warning, '2' error, '3' fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import keras
import tensorflow as tf
from keras.datasets import mnist


def run(modelName):
    if modelName == 'MLP':
        model = bulidMLP()
    elif modelName == 'CNN':
        model = buildCNN()
    model.summary()

    print('\nTraining '+modelName)
    with tf.device('/device:GPU:0'):
        model.fit(xTrain, yTrain, epochs=100, batch_size=1024, validation_split=0.1, 
                  callbacks=keras.callbacks.EarlyStopping(
                      monitor='val_sparse_categorical_accuracy',
                      patience=5, restore_best_weights=True),
                  verbose=3)

    print('\nEvaluating '+modelName)
    res = model.evaluate(xTest, yTest, verbose=0)
    print(model.metrics_names[0],'%.4f'%res[0])
    print(model.metrics_names[1],'%.4f'%res[1], '\n')
    return model


def bulidMLP():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(
            128, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics='sparse_categorical_accuracy')
    return model


def buildCNN():
    model = keras.models.Sequential([
        # keras.layers.Dropout(.4, input_shape=(28,28,1)),
        keras.layers.Conv2D(1, 3, (1, 1), activation='relu', padding='valid', input_shape=(
            28, 28, 1), kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling2D(pool_size=(
            2, 2), strides=(2, 2), padding='valid'),
        keras.layers.Flatten(),
        keras.layers.Dense(
            128, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics='sparse_categorical_accuracy')
    return model


if __name__ == '__main__':
    print('\navailable GPU:', tf.config.list_physical_devices('GPU'))

    print('\nLoading data')
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = xTrain/255.0
    xTest = xTest/255.0

    run('MLP')

    run('CNN')
