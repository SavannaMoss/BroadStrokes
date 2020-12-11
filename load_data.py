# load_data.py
# Contains a method necessary to load data from .gz text files for various algorithms.

import numpy as np
import os

def load_data():
    '''Loads data and labels created from images by image_coversion.py'''

    # read in the training data
    with open("data/train_data.gz", 'r', newline='\n') as row:
        xtrain = np.loadtxt(row, delimiter=",")

    # read in the testing data
    with open("data/test_data.gz", 'r', newline='\n') as row:
        xtest = np.loadtxt(row, delimiter=",")

    # read in labels for training and testing data
    ttrain = np.loadtxt(os.path.realpath("data/train_labels.txt"), dtype='str',  delimiter=",")
    ttest = np.loadtxt(os.path.realpath("data/test_labels.txt"), dtype='str', delimiter=",")

    # return all data and labels
    return (xtrain, ttrain), (xtest, ttest)
