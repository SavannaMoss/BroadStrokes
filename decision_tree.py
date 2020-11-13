# decision_tree.py
"""Predict an artist based on attributes of a given artwork using a decision tree."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code

def main():
    # Relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "data/decision_tree_data", "traindata.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "data/decision_tree_data", "trainlabels.txt"))

    testdatafile = os.path.expanduser(os.path.join(ROOT, "data/decision_tree_data", "testdata.txt"))
    testlabelfile = os.path.expanduser(os.path.join(ROOT, "data/decision_tree_data", "testlabels.txt"))

    attributesfile = os.path.expanduser(os.path.join(ROOT, "data/decision_tree_data", "attributes.txt"))
    classesfile = os.path.expanduser(os.path.join(ROOT, "data/decision_tree_data", "classes.txt"))

    # Load data from relevant files
    print("Loading training data...")
    traindata = np.loadtxt(datafile, str, delimiter=",")
    #print(traindata)
    print("Loading training labels...")
    trainlabels = np.loadtxt(labelfile, str, delimiter=",")

    print("Loading testing data...")
    testdata = np.loadtxt(testdatafile, str, delimiter=",")
    #print(testdata)
    print("Loading testing labels...")
    testlabels = np.loadtxt(testlabelfile, str, delimiter=",")

    print("Loading attributes...")
    attributes = np.loadtxt(attributesfile, str, delimiter=",")

    print("Loading classes...")
    classes = np.loadtxt(classesfile, str, delimiter=",")

    # Train a decision tree via information gain on the training data

    # USE ONEHOT ENCODING FOR CATEGORICAL ATTRIBUTES
    encTrain = OneHotEncoder(handle_unknown='ignore')
    encTrain.fit(traindata)
    encTrain = encTrain.transform(traindata).toarray().astype('float32')


    encTest = OneHotEncoder(handle_unknown='ignore')
    encTest.fit(traindata)
    encTest = encTest.transform(testdata).toarray().astype('float32')



    # create the decision tree
    tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
    tree.fit(encTrain, trainlabels)
    # pdb.set_trace()
    
    #Test the decision tree
    pred = tree.predict(encTest)#, check_input = False)

    # # Show the confusion matrix for test data
    # cm = confusion_matrix(testlabels, pred)
    # print("Confusion matrix:")
    # print(cm)

    # Compare training and test accuracy
    trainaccuracy = tree.score(encTrain, trainlabels)
    print("Training accuracy:", trainaccuracy)

    testaccuracy = tree.score(encTest, testlabels)
    print("Testing accuracy:", testaccuracy)

    # Visualize the tree using matplotlib and plot_tree
    fig = plt.figure(figsize=(13,8))
    fig = plot_tree(tree, feature_names=attributes, class_names= classes, filled=True, rounded=True)
    plt.show()

if __name__ == '__main__':
    main()
