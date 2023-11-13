import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0' info, '1' warning, '2' error, '3' fatal}
import tensorflow as tf
import keras
from matplotlib import pyplot as plt


xTrain, yTrain, xTest, yTest = None, None, None, None
labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
          4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


def printShow(n):
    print(xTrain[n])

    print(labels[yTrain[n][0]])

    plt.imshow(xTrain[n])
    plt.show()


if __name__ == '__main__':
    print('\navailable GPU:', tf.config.list_physical_devices('GPU'))
    print('\nLoading data\n')
    (xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()
    # printShow(1)
    xTrain = xTrain/255
    xTest = xTest/255
    # printShow(1)

    cnn = keras.Sequential([
        keras.layers.Conv2D(9, (3, 3), (1, 1), padding='valid', input_shape=(
            32, 32, 3), activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.AveragePooling2D((9, 9), (1, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            1000, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics="sparse_categorical_accuracy")
    cnn.summary()

    print('\nTraining\n')
    with tf.device('/device:GPU:0'):
        cnn.fit(xTrain, yTrain, epochs=100, batch_size=512, shuffle=True, validation_split=0.1, callbacks=keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy', patience=5, restore_best_weights=True), verbose=1)

    print('\nResults\n')
    res = cnn.evaluate(xTest, yTest, verbose=0)
    print(cnn.metrics_names[0], res[0])
    print(cnn.metrics_names[1], res[1], '\n')


"""
    0,6000 %
    cnn = keras.Sequential([
        keras.layers.Conv2D(9, (3, 3), (1, 1), padding='valid', input_shape=(32, 32, 3),
                            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.AveragePooling2D((9, 9), (9, 9)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            150, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax')
    ])

    0,6000 %
    cnn = keras.Sequential([
        keras.layers.Conv2D(9, (3, 3), (1, 1), padding='valid', input_shape=(
            32, 32, 3), activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.AveragePooling2D((9, 9), (1, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            1000, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax')
    ])


"""
