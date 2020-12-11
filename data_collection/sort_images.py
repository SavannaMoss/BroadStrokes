# sort_images.py
# Utilized to sort images into respective training and testing data subfolders.

import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():

    # load text files
    train = np.loadtxt("textfiles/train_file_names.txt", str, delimiter="/n")
    test = np.loadtxt("textfiles/test_file_names.txt", str, delimiter="/n")

    # establish src and dst
    src = os.path.realpath(os.path.dirname(dir_path)+"\\data\\image_data")
    dst_train = os.path.realpath(os.path.dirname(dir_path)+"\\data\\image_data\\train_images")
    dst_test = os.path.realpath(os.path.dirname(dir_path)+"\\data\\image_data\\test_images")

    search through folder and compare file name to train and test
    for root, dirs, files in os.walk(src):
        for file in files:
            # if (file.endswith('.jpg')) and (file in train):
            #     os.rename(src+"\\"+file, dst_train+"\\"+file)
            #
            # if (file.endswith('.jpg')) and (file in test):
            #     os.rename(src+"\\"+file, dst_test+"\\"+file)

            pass

if __name__ == '__main__':
    main()
