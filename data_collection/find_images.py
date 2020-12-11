# find_images.py
# Utilized to remove unneeded image files from original Kaggle dataset.

import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():

    # load text file
    needed_files = np.loadtxt("textfiles/image_file_names.txt", str, delimiter="/n")

    # establish src
    src = os.path.realpath(os.path.dirname(dir_path)+"\\data\\image_data")
    import pdb; pdb.set_trace()
    # search through folder and compare file name to needed_files
    for root, dirs, files in os.walk(src):
        for file in files:
            if (file.endswith('.jpg')) and (file not in needed_files):
                # os.remove(file)
                
                pass

if __name__ == '__main__':
    main()
