#
#   image_conversion.py
#   Converts a folder of .pngs to a single .txt file
#

import numpy as np
import os
import tensorflow as tf
tf.random.set_seed(3520)

DATAPATH = os.path.realpath('data/image_data')

def main():
    print("Loading data...")

    x = loaddata('train')
    print("Train data file created!")

    xtest = loaddata('test')
    print("Test data file created!")

def loaddata(mode):
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

            # write image array to file
            np.savetxt(f, img_array, newline=",")

            f.write("\n")

            break # temp while testing, remove when using on all data

    # close file
    f.close()

    # remove last comma at the end of the file
    if mode == 'train':
        with open("train_data.txt", 'rb+') as f:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 3, os.SEEK_SET)
            f.truncate()

    elif mode == 'test':
        with open("test_data.txt", 'rb+') as f:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 3, os.SEEK_SET)
            f.truncate()
    else:
        print("Unrecognized mode.")

    # close file
    f.close()

if __name__ == '__main__':
    main()
