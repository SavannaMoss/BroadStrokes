#   image_conversion.py
#   Creates a .txt file from a folder of .pngs

import numpy as np
import os
import tensorflow as tf

DATAPATH = os.path.realpath('data/image_data')

def main():
    print("Loading data...")

    x = createFile('train')
    print("Train data file created!")

    xtest = createFile('test')
    print("Test data file created!")

def createFile(mode):
    sz = 224 # new image size / number of total pixels

    # find correct images
    if mode == 'train':
        img_path = DATAPATH+"\\train_resized"
        f = open("train_data.txt", "a")

    elif mode == 'test':
        img_path = DATAPATH+"\\test_resized"
        f = open("test_data.txt", "a")
    else:
        print("Unrecognized mode.")

    # obtain file names
    files = [f for f in os.listdir(img_path)]

    # loop through files and convert to array
    count = 0
    images = []
    for file in files:
        if (file.endswith('.png')):

            img = tf.keras.preprocessing.image.load_img(
                img_path+"\\"+file,
                color_mode="grayscale",
                target_size=(sz, sz),
                interpolation='bilinear')
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255

            img_array = np.array(img_array).flatten()

            images.append(img_array)

            count += 1
            if(count == 20):
                break # temp while testing, remove when using on all data

    # converting the list of images to an ndarray
    images = np.asarray(images)
    np.savetxt(f, images, delimiter=',', newline='\n')

    # close file
    f.close()

if __name__ == '__main__':
    main()
