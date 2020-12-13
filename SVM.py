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
    '''
    VARIABLE KEY:
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest

    HOG train: x
    HOG test: xt

    Misclassified LabeLs: mislabelled_(train/test)

    '''


    #set seed for repeatability
    np.random.seed(0)

    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    #reshape testing and training for HOG
    xtrain = xtrain.reshape(xtrain.shape[0], 224, 224)
    xtest = xtest.reshape(xtest.shape[0], 224, 224)

    #grab HOG features of training and testing data
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
    print("\nStatistics: ")
    print("Training Accuracy: ", clf.score(x, t))
    print("Testing Accuracy:", clf.score(xt, tt))
    print("Precision:", precision_score(tt, clf.predict(xt), labels = labels, average='micro'))

    misllabelled_train = np.where(clf.predict(x) != t)
    mislabelled_test = np.where(clf.predict(xt) != tt)
    print("Number of Incorrectly Predicted: ", (xtrain[misllabelled_train].shape[0] + xtest[mislabelled_test].shape[0]), " / " , (xtrain.shape[0] + xtest.shape[0]), " images")
    print("Number of Correctly Predicted: ", ((xtrain.shape[0] - xtrain[misllabelled_train].shape[0]) + (xtest.shape[0] - xtest[mislabelled_test].shape[0])), " / " , (xtrain.shape[0] + xtest.shape[0]), " images")

    print("\nCreating Confusion Matrix... ")
    #plot confision matrix
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

#gets the HOG features
def getHOG(data):
    x = []
    for d in data:
        x.append(hog(d, orientations = 45, pixels_per_cell = (16, 16),
                    cells_per_block = (14, 14), transform_sqrt = True))

    return np.asarray(x)

if __name__ == '__main__':
    main()
