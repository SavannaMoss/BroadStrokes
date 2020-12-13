# KNN.py
# Implements a k-nearest-neighbors model to determine an artist based on an artwork.

import numpy as np
import os
import matplotlib.pyplot as plt

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
    Training Data: xtrain
    Training Labels: ttrain
    Testing Data: xtest
    Testing Labels: ttest
    '''

    # dimension reduction for generality
    dim_reduction = NeighborhoodComponentsAnalysis(n_components=80, init='pca', random_state=0)
    dim_reduction.fit(xtrain, ttrain)

    print("Training KNN model...")
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(dim_reduction.transform(xtrain), ttrain)

    # show model
    transformed = dim_reduction.transform(xtest)
    labels, num = np.unique(ttest, return_inverse = True)

    fig, ax = plt.subplots()
    ax.set_title("KNN Scatter Plot")
    scatter = ax.scatter(transformed[:,0], transformed[:,1], c=num, cmap='tab10')
    handles, _ = scatter.legend_elements()
    legend = ax.legend(handles, labels,
                        title='Artists', fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.add_artist(legend)

    plt.show()

    # performance metrics
    print("Testing Accuracy:", clf.score(transformed, ttest))

    print("Precision:", precision_score(ttest, clf.predict(transformed), average='micro'))

    plt.rc('font', size=6)
    plt.rc('figure', titlesize=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2, top=0.9, right=0.9, left=0.1)
    ax.set_title("KNN Confusion Matrix")
    cm = plot_confusion_matrix(clf, transformed, ttest,
                                normalize='all',
                                display_labels=np.unique(ttest),
                                xticks_rotation='vertical',
                                cmap=plt.cm.Blues,
                                ax=ax)
    plt.show()

if __name__ == '__main__':
    main()
