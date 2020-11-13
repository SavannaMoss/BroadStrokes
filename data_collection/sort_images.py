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

    """
    src = "C:/Users/savan/Documents/GitHub/BroadStrokes/data/image_data/"
    dst_train = "C:/Users/savan/Documents/GitHub/BroadStrokes/data/image_data/train_images/"
    dst_test = "C:/Users/savan/Documents/GitHub/BroadStrokes/data/image_data/test_images/"

    # search through folder and compare file name to train and test
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if (file.endswith('.jpg')) and (file in train):
                os.rename(src+file, dst_train+file)

            if (file.endswith('.jpg')) and (file in test):
                os.rename(src+file, dst_test+file)
    """

if __name__ == '__main__':
    main()
