#
#   image_conversion.py
#   Converts a folder of .pngs to a single .txt file
#

import numpy as np
import os
import tensorflow as tf
tf.random.set_seed(3520)

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    # Load images and labels
    print("Loading data...")

    # x = loaddata('train')
    print("Train data file created!")

    xtest = loaddata('test')
    print("Test data file created!")

def loaddata(mode):
    '''Load data and labels.'''
    sz = 224

    if mode == 'train':
        img_path = os.path.realpath("data/image_data/train_resized/")
        f = open("train_data.txt", "a")
    elif mode == 'test':
        img_path = os.path.realpath("data/image_data/test_resized/")
        f = open("test_data.txt", "a")
    else:
        print("Unrecognized mode.")

    count = 0
    for root, dirs, files in os.walk(img_path):
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

                # import pdb; pdb.set_trace()
                np.savetxt(f, img_array, newline=",")
                f.write("\n")
                print("printed new line")

                count += 1
                print(count)

            if count == 3:
                return dir_path
                

    f.close()

if __name__ == '__main__':
    main()
