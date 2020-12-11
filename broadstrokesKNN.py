#broadstrokesKNN.py

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from skimage.util import montage
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
#from image_conversion import loaddata

#set seed number
SEED = 3520
numclasses = 10
k = 1

#get direct path
dir_path = os.path.dirname(os.path.realpath(__file__))


def main():

    print("Loading Training and Testing Data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    #only because we're testing 2 samples
    ttrain, ttest = ttrain[:20], ttest[:20]

    print("Training Model...")
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(xtrain, ttrain)

    print("Accuracy: ", clf.score(xtest, ttest))
    print("Precision:")
    print("Confusion Matrix: ")

    print("Graphing...")




def load_data():
    '''Loads training and testind data and labels made in image_coversion.py'''

    #get the path of the training data
    with open("train_data.txt", 'r', newline='\n') as row:
        xtrain = np.loadtxt(row, delimiter=",")

    #get the testing data
    with open("test_data.txt", 'r', newline='\n') as row:
        xtest = np.loadtxt(row, delimiter=",")

    #grab labels for training and testing data
    ttrain = np.loadtxt(os.path.realpath("data/trainlabels.txt"), dtype='str',  delimiter=",")
    ttest = np.loadtxt(os.path.realpath("data/testlabels.txt"), dtype='str', delimiter=",")


    #return all data and labels
    return (xtrain, ttrain), (xtest, ttest)


if __name__ == '__main__':
    main()
