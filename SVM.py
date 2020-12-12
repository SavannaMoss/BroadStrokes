# broadstrokesSVM.py
# Implements a support vector machine to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn import svm
from skimage.feature import hog
from skimage.util import montage

from sklearn.preprocessing import OneHotEncoder

# custom library
from load_data import load_data

def main():
    numimages = 50
    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    xtrain, ttrain = xtrain[:numimages], ttrain[:numimages]
    xtest, ttest = xtest[:numimages], ttest[:numimages]

    xtrain = xtrain.reshape(numimages, 224, 224)
    xtest = xtest.reshape(numimages, 224, 224)
    #test if the image is working
    # ttemp  = xtrain[0].reshape(224,224)
    # plt.imshow(ttemp)
    # plt.show()

    '''
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''

    # print("Extracting HOG Features...")
    x = getHOG(xtrain)
    xt = getHOG(xtest)
    import pdb; pdb.set_trace()

    print("Setting up labels...")
    # converts categorical class labels to unique values onehot
    names, t = np.unique(ttrain, return_inverse=True)
    names1, tt = np.unique(ttest, return_inverse=True)
    # t = OneHotEncoder(ttrain)
    # tt = OneHotEncoder(ttest)

    print("Training SVM...")
    clf = svm.SVC(C = 0.1)
    # clf.fit(xtrain, t)
    clf.fit(x, t) # for use with HOG features



    print("Testing Accuracy: ", clf.score(xt, tt))
    print("Precision: ")
    print("Confusion Matrix: ")

def getHOG(data):
    x = []
    for d in data:
        x.append(hog(d, orientations = 8, pixels_per_cell = (64, 64),
                    cells_per_block = (1, 1), transform_sqrt = True))

    return np.asarray(x)

if __name__ == '__main__':
    main()
