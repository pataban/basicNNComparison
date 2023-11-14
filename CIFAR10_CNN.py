import os
# or any {'0' info, '1' warning, '2' error, '3' fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from matplotlib import pyplot as plt
import tensorflow as tf
import keras


xTrain, yTrain, xTest, yTest = None, None, None, None
labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
          4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
FILTER = 9
KERNEL = 3
STRIDES = 1
POOL_SIZE = 9
POOL_STRIDE = 1
DENSE_COUNT = 512
"""
POOL_STRIDE=9
DENSE_COUNT=150
"""


def printShow(n):
    print(xTrain[n])

    print(labels[yTrain[n][0]])

    plt.imshow(xTrain[n])
    plt.show()


if __name__ == '__main__':
    print('\navailable GPU:', tf.config.list_physical_devices('GPU'))

    print('\nLoading data')
    (xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()
    # printShow(1)
    xTrain = xTrain/255
    xTest = xTest/255
    # printShow(1)

    model = keras.Sequential([
        keras.layers.Conv2D(FILTER, KERNEL, STRIDES, padding='valid',
                            input_shape=(32, 32, 3), activation='relu',
                            kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.AveragePooling2D(POOL_SIZE, POOL_STRIDE),
        keras.layers.Flatten(),
        keras.layers.Dense(DENSE_COUNT, activation='relu',
                           kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics="sparse_categorical_accuracy")
    model.summary()

    print('\nTraining')
    with tf.device('/device:GPU:0'):
        model.fit(xTrain, yTrain, epochs=100, batch_size=512, validation_split=0.1,
                  callbacks=keras.callbacks.EarlyStopping(
                      monitor='val_sparse_categorical_accuracy',
                      patience=5, restore_best_weights=True),
                  verbose=1)

    print('\nEvaluating')
    res = model.evaluate(xTest, yTest, verbose=0)
    print(model.metrics_names[0], '%.4f' % res[0])
    print(model.metrics_names[1], '%.4f' % res[1], '\n')
