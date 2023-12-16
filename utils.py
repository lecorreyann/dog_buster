import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array_keras, array_to_img
import matplotlib.pyplot as plt

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
def load_img(path):
    '''
    Return an image
    '''
    img = Image.open(path)
    return img

# resize img


def resize_img(img, height=255, width=255, channels=3):
    '''
    Return a resized image
    '''
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
