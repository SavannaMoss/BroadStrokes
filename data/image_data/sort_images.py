import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    # import pdb; pdb.set_trace()

    # load text file
    train_file_names = "train_file_names.txt"
    test_file_names = "test_file_names.txt"
    train = np.loadtxt(train_file_names, str, delimiter="/n")
    test = np.loadtxt(test_file_names, str, delimiter="/n")

    dst_train = dir_path+"\\train_images"
    dst_test = dir_path+"\\test_images"

    search through folder and compare file name to train and test
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if (file.endswith('.jpg')) and (file in train):
                os.rename(dir_path, dst_train)

            if (file.endswith('.jpg')) and (file in test):
                os.rename(dir_path, dst_test)

if __name__ == '__main__':
    main()
