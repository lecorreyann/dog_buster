# balance datasets
from utils import img_to_array
from data_transformation import data_transformation
import os
import numpy as np
import time

# Balance datasets


def balance_datasets(dir_train_0=os.environ.get('DOGS_DATASET_TRAIN_PATH') + '/0', dir_train_1=os.environ.get('DOGS_DATASET_TRAIN_PATH') + '/1'):
    '''
    Balance datasets with data augmentation
    '''
    # print the message with picture icon utf8
    print('\U0001F5BC Balance datasets')
    # Get the list of images path
    list_img_path_0 = os.listdir(dir_train_0)
    list_img_path_1 = os.listdir(dir_train_1)
    # Get the len of the directory with most images
    max_len = max(len(list_img_path_0), len(list_img_path_1))
    dir_less_img = ''
    # Get the dataset with less images
    if len(list_img_path_0) < len(list_img_path_1):
        dir_less_img = dir_train_0
    else:
        dir_less_img = dir_train_1

    if len(os.listdir(dir_less_img)) < max_len:
        for i in range(max_len - len(os.listdir(dir_less_img))):
            # get random number bewteen 0 and len of dir -1
            random_img_index = np.random.randint(
                0, len(os.listdir(dir_less_img))-1)
            print(random_img_index)
            # wait before next iteration
            data_transformation(img_array=img_to_array(
                dir_less_img + '/' + os.listdir(dir_less_img)[random_img_index]), output_dir=dir_less_img, qty=1)
            time.sleep(1)
