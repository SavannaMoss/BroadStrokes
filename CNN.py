# CNN.py
# Implements a convolutional neural network to determine an artist based on an artwork.

import numpy as np
import os
import math
import random
import matplotlib.pyplot as plt
import pdb

# tensorflow imports
import tensorflow as tf
tf.random.set_seed(3520)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD

# sci-kit imports
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# custom library
from load_data import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code

def main(custom=None, skip_load=None):

    if custom == None:
        # custom is used to determine the params of inputs for layers
        custom = [32, 15, 3, 16, 15, 3, 2000, 800, 200, 0.005]

    # skip load used to ignore loading the data when we already have it loaded
    if skip_load == None:
        print("Loading training and testing data...")
        (xtrain, ttrain), (xtest, ttest) = load_data()
        print("Load data complete.")
    else:
        xtrain = skip_load[0]
        ttrain = skip_load[1]
        xtest = skip_load[2]
        ttest = skip_load[3]

    # normalize the data
    trainnumsamples = xtrain.shape[0]
    testnumsamples = xtest.shape[0]
    px = xtrain.shape[1]
    sz = int(math.sqrt(px))
    xtrain = xtrain.reshape(trainnumsamples, sz, sz, 1)
    xtest = xtest.reshape(testnumsamples, sz, sz, 1)

    # define useful variables
    numsamples, xtrainxaxis, xtrainyaxis, _ = xtrain.shape
    numinputs = xtrainxaxis * xtrainyaxis
    numoutputs = len(np.unique(ttrain))

    # convert output to categorical targets
    print("Converting labels to onehot encoding...")
    enc = OneHotEncoder()
    enc.fit(np.unique(ttrain).reshape(-1, 1))
    ttrainOH = enc.transform(ttrain.reshape(-1, 1)).toarray()
    ttestOH = enc.transform(ttest.reshape(-1, 1)).toarray()

    # create the network
    print("Creating the network...")
    print("Layer Parameters:", custom)
    model = Sequential()

    # add convolutional layer
    model.add(Conv2D(
            filters=custom[0],
            kernel_size=(custom[1], custom[1]),
            activation='relu',
            input_shape=(xtrainxaxis, xtrainyaxis, 1),
            name='Conv1'
            )
        )

    # add max pooling layer
    model.add(MaxPool2D(
            pool_size=(custom[2], custom[2]),
            name='Pooling1'
            )
        )

    # add convolutional layer
    model.add(Conv2D(
            filters=custom[3],
            kernel_size=(custom[4], custom[4]),
            activation='relu',
            name='Conv2'
            ),
        )

    # add max pooling layer
    model.add(MaxPool2D(
            pool_size=(custom[5], custom[5]),
            name='Pooling2'
            )
        )

    # flatten data for dense layer
    model.add(Flatten())

    # add dense layers
    model.add(Dense(units=custom[6], activation='relu', name='dense1'))
    model.add(Dense(units=custom[7], activation='relu', name='dense2'))
    model.add(Dense(units=custom[8], activation='relu', name='dense3'))
    model.add(Dense(units=numoutputs, activation='softmax', name='output'))
    model.summary()

    # Compile the network
    model.compile(
        loss='mse',
        optimizer=SGD(learning_rate=custom[9]),
        metrics=['accuracy'])

    print("Training the network...")
    history = model.fit(xtrain, ttrainOH,
                        epochs=20,
                        batch_size=30,
                        verbose=0,
                        validation_data=(xtest, ttestOH))

    # save the model
    print("Saving model...")
    model.save(os.path.expanduser(os.path.join(ROOT, 'CNN_model.h5')))
    print("Model saved!")

    print("=================================")
    # performance metrics
    print("\nComputing performance metrics...")

    train_metrics = model.evaluate(xtrain, ttrainOH, verbose=0) # training accuracy
    print(f"Training Loss: {train_metrics[0]:0.4f}")
    print(f"Training Accuracy: {train_metrics[1]:0.4f}")

    test_metrics = model.evaluate(xtest, ttestOH, verbose=0) # testing accuracy
    print(f"Testing Loss: {test_metrics[0]:0.4f}")
    print(f"Testing Accuracy: {test_metrics[1]:0.4f}")

    # display plot
    print("Displaying plot...")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    return metrics[1], model

if __name__ == '__main__':
    main()
