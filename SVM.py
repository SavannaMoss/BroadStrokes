# broadstrokesSVM.py
# Implements a support vector machine to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn import svm
from skimage.feature import hog
from skimage.util import montage

# custom library
from load_data import load_data

def main():

    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    xtrain, ttrain = xtrain[:20], ttrain[:20]
    xtest, ttest = xtest[:20], ttest[:20]

    '''
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''

    print("Extracting HOG Features...")
    x = getHOG(xtrain)
    xt = getHOG(xtest)

    print("Setting up labels...")
    # converts categorical class labels to unique values 0-9
    names, t = np.unique(ttrain, return_inverse=True)
    names1, tt = np.unique(ttest, return_inverse=True)

    print("Training SVM...")
    clf = svm.SVC(C = 0.1)
    # clf.fit(xtrain, ttrain)
    clf.fit(x, xt) # for use with HOG features

    import pdb; pdb.set_trace()

    print("Testing Accuracy: ", clf.score(xtest, ttest))
    print("Precision: ")
    print("Confusion Matrix: ")

def getHOG(data):
    x = []
    for d in data:
        x.append(hog(d, orientations = 8, pixels_per_cell = (16, 16),
                    cells_per_block = (1, 1), transform_sqrt = True))

    return np.asarray(x)

if __name__ == '__main__':
    main()
