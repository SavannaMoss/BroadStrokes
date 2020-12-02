# Convolutuonal_Neural_Network.py
# Predicting the artist for a given image
#       Fall 2020, CSC 3520, Group project
#

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import random
from sklearn import datasets
from skimage.util import montage
from sklearn.metrics import confusion_matrix
from image_conversion import loaddata


ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code


def main():
    set_random_seed(3520)  # set random seed

    # LOAD IMAGES AND LABELS
    ''' 
    INSERT CODE HERE
    '''

    # Convert folder of .pngs to .txt file
    if os.path.exists("train_data.txt") == False:   # Check if our file already exists
        xtrainpath = loaddata('train')  # Create file

    # Convert folder of .pngs to .txt file
    if os.path.exists("test_data.txt") == False:   # Check if our file already exists
        xtestpath = loaddata('test')  # Create file

    
    xtrain = np.genfromtxt("train_data.txt", dtype=None, delimiter=",")
    

    #xtest = np.genfromtxt("test_data.txt", dtype=None, delimiter=",")

    dir_path = os.path.dirname("C:/Users/brock/Documents/BroadStrokes/data/")
    ytest = np.genfromtxt(dir_path+"/testlabels.txt", dtype=None, delimiter=",")
    ytrain = np.genfromtxt(dir_path+"/trainlabels.txt", dtype=None, delimiter=",")

    

    # Normalize the data
    #xtrain = xtrain.transpose(2, 0, 1)
    xtrain = xtrain.reshape(xtrain.shape[0], (xtrain.shape[1])//2, (xtrain.shape[1])//2)

    #xtest = xtest.transpose(2, 0, 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2])
    
    
    #show_images(x, x.shape[0])
    #pdb.set_trace()
    numsamples, xtrainxaxis, xtrainyaxis, _ = xtrain.shape
    numinputs = xtrainxaxis * xtrainyaxis
    numoutputs = len(np.unique(ytrain))
    
    # Convert output to categorical targets
    t = tf.keras.utils.to_categorical(ytrain, numoutputs)

    ttest = tf.keras.utils.to_categorical(ytest, numoutputs)


    # Create neural network
    model = Sequential()
    
    # Add convolutuonal layer
    model.add(  Conv2D(
            filters=64, 
            kernel_size=(3, 3),
            activation='relu', 
            input_shape=(xtrainxaxis, xtrainyaxis, 1),
            name='Conv1'
            )
        )
    # Add max pooling layer
    model.add(  MaxPool2D(
            pool_size=(3,3),
            name='Pooling1' 
            )
        )

    # Add convolutuonal layer
    model.add(  Conv2D(
            filters=64, 
            kernel_size=(3, 3),
            activation='relu', 
            name='Conv2'
            ), 
        )
    # Add max pooling layer
    model.add(  MaxPool2D(
            pool_size=(3,3),
            name='Pooling2'
            )
        )

    # Flatten data for dense layer
    model.add(Flatten())

    # Add dense layers
    model.add(Dense(units=30, activation='relu', name='dense1'))
    model.add(Dense(units=30, activation='relu', name='dense2'))
    model.add(Dense(units=numoutputs, activation='softmax', name='output'))
    model.summary()

    
    model.compile(
        loss='mse',
        optimizer=SGD(learning_rate=0.02),
        metrics=['accuracy'])       
    
    # Train the network
    history = model.fit(xtrain, t,
                        epochs=20,
                        batch_size=100,
                        verbose=0,
                        validation_data=(xtest, ttest))
    
    # Compute the training accuracy
    metrics = model.evaluate(xtrain, t, verbose=0)

    # Compute the testing accuracy
    #metrics = model.evaluate(xtest, ttest, verbose=0)
    print("=================================")
    print(f"loss = {metrics[0]:0.4f}")
    print(f"accuracy = {metrics[1]:0.4f}")

    # Display plot
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    # Confusion Matrix

    '''
    INSERT CODE HERE
    '''

    # Save model to file
    model.save(os.path.expanduser(os.path.join(ROOT, 'digits_model.h5')))


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
