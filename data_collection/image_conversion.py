# image_conversion.py
# Creates a .gz text file from a folder of .pngs

import numpy as np
import os
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))

DATAPATH = os.path.dirname(dir_path)+'\\data\\image_data\\'

DST = os.path.dirname(dir_path)+'\\data\\'

def main():
    print("Loading data...")

    x = createFile('train')
    print("Train data file created!")

    xtest = createFile('test')
    print("Test data file created!")

def createFile(mode):
    sz = 224 # new image size / number of total pixels
    count = 0
    # find correct images and open relevent file
    if mode == 'train':
        img_path = DATAPATH+"train_resized"
        f = open(DST+"train_data.gz", "w")

    elif mode == 'test':
        img_path = DATAPATH+"test_resized"
        f = open(DST+"test_data.gz", "w")

    else:
        print("Unrecognized mode.")

    # obtain file names
    files = [f for f in os.listdir(img_path)]

     # intialize images list
    images = []

    # loop through file names
    for file in files:
        if (file.endswith('.png')):

            # convert to 2D array of values between 0 and 1 per pixel
            img = tf.keras.preprocessing.image.load_img(
                img_path+"\\"+file,
                color_mode="grayscale",
                target_size=(sz, sz),
                interpolation='bilinear')
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255

            # flatten to 1D array
            img_array = np.array(img_array).flatten()

            # append image to images list
            images.append(img_array)
            count += 1;
            if (count == 50):
                break

    # convert the list of images to an ndarray
    images = np.asarray(images)

    # save the array to file
    np.savetxt(f, images, delimiter=',', newline='\n')

    # close file
    f.close()

if __name__ == '__main__':
    main()
