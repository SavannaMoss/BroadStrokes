# SVM.py
# Implements a support vector machine to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn import svm
from skimage.feature import hog
from skimage.util import montage
from sklearn.metrics import precision_score, plot_confusion_matrix

# custom library
from load_data import load_data

def main():
    np.random.seed(0)

    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    xtrain = xtrain.reshape(xtrain.shape[0], 224, 224)
    xtest = xtest.reshape(xtest.shape[0], 224, 224)

    '''
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''

    print("Extracting HOG features...")
    x = getHOG(xtrain)
    xt = getHOG(xtest)

    print("Setting up labels...")
    # converts categorical class labels to unique values
    _, t = np.unique(ttrain, return_inverse=True)
    labels, tt = np.unique(ttest, return_inverse=True)

    print("Training SVM...")
    clf = svm.SVC(C=1.0)
    clf.fit(x, t)

    # performance metrics
    print("Testing Accuracy:", clf.score(xt, tt))

    print("Precision:", precision_score(tt, clf.predict(xt), average='micro'))

    plt.rc('font', size=6)
    plt.rc('figure', titlesize=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2, top=0.9, right=0.9, left=0.1)
    ax.set_title("SVM Confusion Matrix")
    cm = plot_confusion_matrix(clf, xt, tt,
                                normalize='all',
                                display_labels=labels,
                                xticks_rotation='vertical',
                                cmap=plt.cm.Blues,
                                ax=ax)
    plt.show()

def getHOG(data):
    x = []
    for d in data:
        x.append(hog(d, orientations = 45, pixels_per_cell = (16, 16),
                    cells_per_block = (14, 14), transform_sqrt = True))

    return np.asarray(x)

if __name__ == '__main__':
    main()
