# CNN.py
# Implements a convolutional neural network to determine an artist based on an artwork.

import numpy as np
import os
import math
import random
import matplotlib.pyplot as plt

# tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD

# sci-kit imports
from sklearn import datasets
from skimage.util import montage
from sklearn.metrics import confusion_matrix

# custom library
from load_data import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code

def main():
    set_random_seed(3520)  # set random seed

    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    '''
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''

    # normalize the data
    numsamples = xtrain.shape[0]
    px = xtrain.shape[1]
    sz = int(math.sqrt(px))
    xtrain = xtrain.reshape(numsamples, sz, sz, 1)

    # show_images(x, x.shape[0])
    numsamples, xtrainxaxis, xtrainyaxis, _ = xtrain.shape
    numinputs = xtrainxaxis * xtrainyaxis
    numoutputs = len(np.unique(ttrain))

    # convert output to categorical targets
    t = tf.keras.utils.to_categorical(ttrain, numoutputs)
    ttest = tf.keras.utils.to_categorical(ttest, numoutputs)

    print("Creating the network...")
    model = Sequential()

    # add convolutional layer
    model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(xtrainxaxis, xtrainyaxis, 1),
            name='Conv1'
            )
        )

    # add max pooling layer
    model.add(MaxPool2D(
            pool_size=(3,3),
            name='Pooling1'
            )
        )

    # add convolutional layer
    model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            name='Conv2'
            ),
        )

    # add max pooling layer
    model.add(MaxPool2D(
            pool_size=(3,3),
            name='Pooling2'
            )
        )

    # flatten data for dense layer
    model.add(Flatten())

    # add dense layers
    model.add(Dense(units=30, activation='relu', name='dense1'))
    model.add(Dense(units=30, activation='relu', name='dense2'))
    model.add(Dense(units=numoutputs, activation='softmax', name='output'))
    model.summary()

    model.compile(
        loss='mse',
        optimizer=SGD(learning_rate=0.02),
        metrics=['accuracy'])

    print("Training the network...")
    history = model.fit(xtrain, t,
                        epochs=20,
                        batch_size=100,
                        verbose=0,
                        validation_data=(xtest, ttest))

    # save model to file
    model.save(os.path.expanduser(os.path.join(ROOT, 'digits_model.h5')))

    # compute performance metrics
    print("=================================")
    train_metrics = model.evaluate(xtrain, t, verbose=0) # training accuracy
    print(f"Training Loss: {train_metrics[0]:0.4f}")
    print(f"Training Accuracy: {train_metrics[1]:0.4f}")

    test_metrics = model.evaluate(xtest, ttest, verbose=0) # testing accuracy
    print(f"Testing Loss: {test_metrics[0]:0.4f}")
    print(f"Testing Accuracy: {test_metrics[1]:0.4f}")

    print("Displaying plot...")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    print("\nCreating confusion matrix... ")
    plt.rc('font', size=6)
    plt.rc('figure', titlesize=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2, top=0.9, right=0.9, left=0.1)

    ax.set_title("CNN Confusion Matrix")
    cm = plot_confusion_matrix(model, xtest, ttest,
                                normalize='all',
                                display_labels=labels,
                                xticks_rotation='vertical',
                                cmap=plt.cm.Blues,
                                ax=ax)
    plt.show()

def set_random_seed(seed):
    '''Set random seed for repeatability.'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def show_images(data, N=1, shape=None):
    '''Show N random samples from the dataset.'''
    numsamples, numpixels = data.shape
    sz = int(numpixels ** (0.5))
    ind = np.random.choice(numsamples, N, replace=False)
    ind.shape = (len(ind),)
    images = data.reshape((numsamples, sz, sz))
    if shape is None:
        s = int(np.ceil(N**(0.5)))
        shape = (s, s)
    m = montage(images[ind, :, :], grid_shape=shape)
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
