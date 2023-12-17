# balance datasets
from utils import get_len_of_directory_with_most_imagen, get_datasets_path, img_to_array
from data_transformation import data_transformation
import os
import numpy as np
import time

# Get the list of images path
max_qt_img = get_len_of_directory_with_most_imagen()

# Get dirs of datasets
dirs_datasets = get_datasets_path()

for dir in dirs_datasets:
    print(dir)
    # get len of dir
    if len(os.listdir(dir)) < max_qt_img:
        for i in range(max_qt_img - len(os.listdir(dir))):
            # get random number bewteen 0 and len of dir -1
            random_img_index = np.random.randint(0, len(os.listdir(dir))-1)
            print(random_img_index)
            # wait before next iteration
            # sleep 2 seconds

            data_transformation(img_array=img_to_array(
                dir + '/' + os.listdir(dir)[random_img_index]), output_dir=dir, qty=1)
            time.sleep(1)
