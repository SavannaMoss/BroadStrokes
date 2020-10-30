import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():

    # load text file
    image_file_names = "image_file_names.txt"
    needed_files = np.loadtxt(image_file_names, str, delimiter="/n")

    # search through folder and compare file name to needed_files
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if (file.endswith('.jpg')) and (file not in needed_files):
                # os.remove(file)

if __name__ == '__main__':
    main()
