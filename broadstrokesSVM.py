#broadstrokesSVM.py
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage.util import montage
from sklearn import svm  # use SVC
# from pandas import Categoical
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    np.random.seed(0)

    print("Loading Training and Testing Data...")
    (xtrain, ttrain), (xtest, ttest) = load_data()

    #only because we're testing 2 samples
    ttrain, ttest = ttrain[:20], ttest[:20]

    # print()
    # print("Extracting HOG Features...")
    # Extract HOG features from each image and concatenate into an array***
    # TIP: Append features to a list in a loop, then convert to an array using np.asarray()
    # x = getHOG(xtrain)
    # xt = getHOG(xtest)
    # import pdb; pdb.set_trace()

    print("Setting up labels...")
    # Create a numpy array of target labels (use 0 for without mask, 1 for with mask)***
    names, t = np.unique(ttrain, return_inverse=True)
    names1, tt = np.unique(ttest, return_inverse=True)


    print("Training SVM...")
    # Train a soft linear support vector machine (C=0.1)*** and fit it to the dataset
    clf = svm.SVC(C = 0.1)
    clf.fit(xtrain, ttrain)

    print("Testing Accuracy: ", clf.score(xtest, ttest))

# def getHOG(data):
#     x = []
#     for d in data:
#         #append all hog of face images to x and append the label 0 to t     cells_per_block = (5,5),\
#         x.append(hog(d, orientations = 8, pixels_per_cell = (100,100), cells_per_block = np.zeros((224, 224)), transform_sqrt = True))
#
#
#     return np.asarray(x)




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
