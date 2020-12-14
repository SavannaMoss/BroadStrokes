# BroadStrokes
### Classifying an Artist’s Impression Using Machine Learning

Developed by Ariyana Miri, Savanna Moss, Hannah Wilberding, and Brock Wilson.

Developed a convolutional neural network, k-nearest-neighbors classifier, and a support vector machine to predict an artist based on images of artworks.

# How to Use the Classifiers
## Generate Data Files
Run *image_conversion.py* to generate the *train_data.gz* and *train_data.gz* compressed text files based on the images in *image_data/train_resized* and *image_data/test_resized*. When loading the files into another program, the train_data ndarray should be of shape (2184, 50176) and the test_data ndarray should be of shape (546, 50176).

## Run Algorithms
*KNN.py* is the k-nearest-neighbors classifier.
- Displays training accuracy, testing accuracy, number of mislabeled and correctly labeled images, a 3D scatter plot of the first three components created by the PCA of the testing data, and a confusion matrix graph.

*SVM.py* is the support vector machine.
- Displays training accuracy, testing accuracy, number of mislabeled and correctly labeled images, and a confusion matrix graph.

*CNN.py* is the convolutional neural network.
- Displays training loss and accuracy, testing loss and accuracy, and a line graph of the training and testing accuracy over time.
