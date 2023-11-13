import keras
from keras.datasets import mnist



def runMLP():
    print("\nrunning MLP")

    model = keras.models.Sequential([ 
            keras.layers.Flatten(input_shape=(28, 28)), 
            keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'), 
            keras.layers.Dense(10, activation='softmax') ]) 
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics="sparse_categorical_accuracy")
    
    print("\ntraining MLP")
    model.fit(trainX, trainY, epochs=30, validation_split=0.1, 
            callbacks=keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, restore_best_weights=True), 
            verbose=2)

    print("\nevaluating MLP")
    model.evaluate(testX, testY, verbose=2)

    print("\nfinished MLP\n")
    return model


def runCNN():
    print("\nrunning CNN")

    model = keras.models.Sequential([ 
            #keras.layers.Dropout(.4, input_shape=(28,28,1)),
            keras.layers.Conv2D( 1, 3,(1,1), activation='relu', padding='valid', input_shape=(28,28,1), kernel_initializer='he_normal', bias_initializer='he_normal'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Flatten(), 
            keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'), 
            keras.layers.Dense(10, activation='softmax') ]) 
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics="sparse_categorical_accuracy")
    
    print("\ntraining CNN")
    model.fit(trainX, trainY, epochs=30, validation_split=0.1, 
            callbacks=keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, restore_best_weights=True), 
            verbose=2)

    print("\nevaluating CNN")
    model.evaluate(testX, testY, verbose=2)

    print("\nfinished CNN\n")
    return model


if __name__=='__main__':
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX=trainX/255.0
    testX=testX/255.0
    trainX=trainX[0:10000]
    trainY=trainY[0:10000]
    testX=testX[0:1000]
    testY=testY[0:1000]
    print("\nData load completed")


    runMLP()


    runCNN()


