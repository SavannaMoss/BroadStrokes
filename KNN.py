# KNN.py
# Implements a k-nearest-neighbors model to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt

# sci-kit imports
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from skimage.util import montage
from sklearn.metrics import precision_score, confusion_matrix

#PCA for dimention reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# custom library
from load_data import load_data

def main():
    n_neighbors = 28
    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    '''
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''
    dim_reduction = NeighborhoodComponentsAnalysis(n_components = 64, init = 'pca') #it will choose beseet one pca, lda,identity, random, nparray

    print("Training KNN model...")
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    #clf.fit(xtrain, ttrain)


    dim_reduction.fit(xtrain, ttrain)
    clf.fit(dim_reduction.transform(xtrain), ttrain)
    # print("Training Accuracy", name, ": ", clf.score(model.trainsform(xtrain), ttrain))
    print("Testing Accuracy: ", clf.score(dim_reduction.transform(xtest), ttest))





    # compute performance metrics
    # print("Training Accuracy:", clf.score(xtrain, ttrain))
    # print("Testing Accuracy:", clf.score(xtest, ttest))


    # pred = clf.predict(xtest)
    # print("Precision:", precision_score(ttest, pred, average='micro'))
    # print("Confusion Matrix:\n", confusion_matrix(ttest, pred))

if __name__ == '__main__':
    main()
