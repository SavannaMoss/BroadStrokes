# load_data.py
# Contains a method necessary to load data from textfiles for various algorithms.

import numpy as np
import os

def load_data():
    '''Loads data and labels created from images by image_coversion.py'''

    # read in the training data
    with open("train_data.txt", 'r', newline='\n') as row:
        xtrain = np.loadtxt(row, delimiter=",")

    # read in the testing data
    with open("test_data.txt", 'r', newline='\n') as row:
        xtest = np.loadtxt(row, delimiter=",")

    # read in labels for training and testing data
    ttrain = np.loadtxt(os.path.realpath("data/trainlabels.txt"), dtype='str',  delimiter=",")
    ttest = np.loadtxt(os.path.realpath("data/testlabels.txt"), dtype='str', delimiter=",")

    # return all data and labels
    return (xtrain, ttrain), (xtest, ttest)
