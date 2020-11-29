# neural_network.py

import numpy as np
import os
import tensorflow as tf
tf.random.set_seed(3520)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    # Load images and labels
    print("Loading data...")
    x, y = loaddata('train')
    xtest, ytest = loaddata('test')

    numsamples, numinputs = x.shape
    numoutputs = len(np.unique(y))

    # Convert to categorical targets
    t = tf.keras.utils.to_categorical(y, numoutputs)
    ttest = tf.keras.utils.to_categorical(ytest, numoutputs)

    # Create network
    print("Creating neural network...")
    model = Sequential()
    model.add(Dense(units=300, activation='relu', name='hidden1', input_shape=(numinputs,)))
    model.add(Dense(units=300, activation='relu', name='hidden2'))
    model.add(Dense(units=numoutputs, activation='softmax', name='output'))

    model.compile(loss='mse',
        optimizer=SGD(learning_rate=0.02),
        metrics=['accuracy'])

    # Train the network
    print("Training...")
    history = model.fit(x, t, epochs=100, batch_size=100, verbose=0, validation_data=(xtest, ttest))
    model.summary()

    train_metrics = model.evaluate(x, t, verbose=0)
    print(f"train loss = {train_metrics[0]:0.3f}")
    print(f"train accuracy = {train_metrics[1]:0.3f}")

    test_metrics = model.evaluate(xtest, ttest, verbose=0)
    print(f"test loss = {test_metrics[0]:0.3f}")
    print(f"test accuracy = {test_metrics[1]:0.3f}")

    pred = model.predict(xtest, verbose=0)

    # show confusion matrix with test data
    cm = confusion_matrix(ytest, np.argmax(pred, axis=1))
    print("Confusion matrix:")
    print(cm)

    plotnetwork(history)

def loaddata(mode):
    '''Load data and labels.'''
    sz = 1300

    if mode == 'train':
        img_path = os.path.realpath("data/image_data/train_resized")

        t = np.loadtxt('data/trainlabels.txt', str, delimiter="/n")

        for root, dirs, files in os.walk(dir_path+img_path):
            for file in files:
                if (file.endswith('.png')):

                    img = tf.keras.preprocessing.image.load_img(
                        file,
                        target_size=(sz, sz),
                        interpolation='bilinear')
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    img_array = img_array / 255

                    data = np.concatenate(data, img_array)

                    import pdb; pdb.set_trace()

        # t = np.loadtxt('data/trainlabels.txt', str)

    elif mode == 'test':
        img_path = os.path.realpath("data/image_data/test_resized")
        data = np.array()

        for root, dirs, files in os.walk(dir_path+img_path):
            for file in files:
                if (file.endswith('.png')):

                    img = tf.keras.preprocessing.image.load_img(
                        file,
                        target_size=(sz, sz),
                        interpolation='bilinear')
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    img_array = img_array / 255

                    data = np.concatenate(data, img_array)

        t = np.loadtxt('data/testlabels.txt', str, delimiter="/n")

    else:
        print("Unrecognized mode.")

    x = data.reshape((data.shape[0], sz*sz))

    return x, t

def plotnetwork(history):
    fig = plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.legend(['train', 'test'], loc='lower right')

    plt.show()

if __name__ == '__main__':
    main()
