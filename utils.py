import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array_keras, array_to_img
import matplotlib.pyplot as plt
import cv2

# Return a list of images path from a directory


def load_img_path_from_dir(dir_path):
    '''
    Return a list of images path from a directory
    '''
    imgs_path = []
    for filename in os.listdir(dir_path):
        imgs_path.append(dir_path + '/' + filename)
    return imgs_path


# load img
def load_img(path, width=255, height=255):
    '''
    Return an image
    '''
    # Load the image
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    return img

# resize img


def resize_img(img, height=255, width=255, channels=3):
    '''
    Return a resized image
    '''

    # Resize the image
    img = np.resize(img, (height, width, channels))
    return img

# reshape img


def reshape_img(img, height=255, width=255, channels=3):
    '''
    Return a reshaped image
    '''
    img = img.reshape((-1, height, width, channels))
    return img

# convert img to array


def img_to_array(img_path):
    '''
    Return a numpy array of an image
    '''
    img = load_img(img_path)
    img = img_to_array_keras(img)
    img = np.expand_dims(img, axis=0)
    img = resize_img(img)
    img = reshape_img(img)
    # display the image
    # quit extra dimension
    # plt.imshow(array_to_img(img[0].astype(np.uint8)))
    # plt.show()
    return img

# Get the list of children datasets path


def get_datasets_path():
    '''
    Return a list of children datasets path
    '''
    root_directory = os.environ.get('DATASETS_PATH')
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        depth = dirpath[len(root_directory) +
                        len(os.path.sep):].count(os.path.sep)
        if depth == 0:
            for dirname in dirnames:
                dir_full_path = os.path.join(dirpath, dirname)
                depth = dir_full_path[len(
                    root_directory) + len(os.path.sep):].count(os.path.sep)
                if depth == 1:
                    dir_relative_path = os.path.relpath(
                        dir_full_path, root_directory)
                    dirs.append(root_directory + '/' + dir_relative_path)
    return dirs

# Get the length of each directory


def get_dirs_len(dirs):
    '''
    Return a list of the length of each directory
    '''
    return [len(load_img_path_from_dir(dir)) for dir in dirs]

# Get the length of the directory with the most images


def get_len_of_directory_with_most_imagen():
    '''
    Return the length of the directory with the most images
    '''
    dirs = get_datasets_path()
    dirs_len = get_dirs_len(dirs)
    return max(dirs_len)
