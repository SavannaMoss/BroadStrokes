# KNN.py
# Implements a k-nearest-neighbors model to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# sci-kit imports
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from skimage.util import montage
from sklearn.metrics import precision_score, plot_confusion_matrix

# custom library
from load_data import load_data

def main():
    n_neighbors = 28
    print("Loading training and testing data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    '''
    VARIABLE NAME KEY:

    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest

    Dimension Reductionn: pca
    Transformed data via pca: pca_(train/test)

    Predicted Labels: pred_test
    Misclassified Test LabeLs: mislabeled_test
    '''

    # dimension reduction using principal component analysis (PCA)
    pca = NeighborhoodComponentsAnalysis(n_components=80, init='pca', random_state=0)
    pca.fit(xtrain, ttrain)

    # transform training and testing data
    pca_train = pca.transform(xtrain)
    pca_test = pca.transform(xtest)

    # converts categorical class labels to unique values
    labels, num = np.unique(ttest, return_inverse=True)

    # training the KNN with pca transformed data
    print("Training KNN model...")
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(pca_train, ttrain)

    # get predicted labels for testing (training yields 100% accuracy so not needed)
    pred_test = clf.predict(pca_test)

    # performance metrics
    print("\nComputing performance metrics...: ")
    print("Training Accuracy:", clf.score(pca_train, ttrain))
    print("Testing Accuracy:", clf.score(pca_test, ttest))

    # count mislabeled data (training yields 100% accuracy so not needed)
    mislabeled_test = np.where(pred_test != ttest)
    print("Number of Incorrectly Predicted:", (xtest[mislabeled_test].shape[0]), "/" ,
                                                (xtrain.shape[0] + xtest.shape[0]), "images")
    print("Number of Correctly Predicted:", (xtrain.shape[0] + (xtest.shape[0] - xtest[mislabeled_test].shape[0])), "/" ,
                                                (xtrain.shape[0] + xtest.shape[0]), "images")

    # plot out dataset with the true labels to see overlaps
    print("\nCreating scatter plot...")
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='3d'))
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.7, left=0.1)

    ax.set_title("KNN Scatter Plot")
    scatter = ax.scatter(pca_test[:,0], pca_test[:,1], pca_test[:,2], c=num, cmap='tab10')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    handles, _ = scatter.legend_elements()
    legend = ax.legend(handles, labels,
                        title='Artists',
                        fontsize='small',
                        bbox_to_anchor=(1.10, 1),
                        loc='upper left',
                        borderaxespad=0)
    ax.add_artist(legend)
    plt.show()

    print("Creating confusion matrix...")
    plt.rc('font', size=6)
    plt.rc('figure', titlesize=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2, top=0.9, right=0.9, left=0.1)

    ax.set_title("KNN Confusion Matrix")
    cm = plot_confusion_matrix(clf, pca_test, ttest,
                                normalize='all',
                                display_labels=np.unique(ttest),
                                xticks_rotation='vertical',
                                cmap=plt.cm.Blues,
                                ax=ax)
    plt.show()

if __name__ == '__main__':
    main()
