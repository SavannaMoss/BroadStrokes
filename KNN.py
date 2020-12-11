# KNN.py
# Implements a k-nearest-neighbors model to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn.neighbors import KNeighborsClassifier
from skimage.util import montage

# custom library
from load_data import load_data

def main():

    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    '''
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''

    # temporary for testing 20 samples
    ttrain, ttest = ttrain[:20], ttest[:20]

    print("Training KNN model...")
    clf = KNeighborsClassifier(3, weights='distance')
    clf.fit(xtrain, ttrain)

    # compute performance metrics
    print("Testing Accuracy: ", clf.score(xtest, ttest))
    print("Precision: ")
    print("Confusion Matrix: ")

if __name__ == '__main__':
    main()
