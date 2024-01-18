import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array_keras, array_to_img
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import zipfile
import gdown
import time
from cloudinary import CloudinaryImage, uploader, config
import requests
from io import BytesIO
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
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
def load_img(path, width=int(os.environ.get('WIDTH')), height=int(os.environ.get('HEIGHT'))):
    '''
    Return an image
    '''
    # Load the image
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    return img

# resize img


def resize_img(img, height=int(os.environ.get('HEIGHT')), width=int(os.environ.get('WIDTH')), channels=3):
    '''
    Return a resized image
    '''

    # Resize the image
    img = np.resize(img, (height, width, channels))
    return img

# reshape img


def reshape_img(img, height=int(os.environ.get('HEIGHT')), width=int(os.environ.get('WIDTH')), channels=3):
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

def getImage(url):
    '''
    Get an image from an url
    '''
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def plot_history(history, title='', axs=None, exp_name="", metric='accuracy'):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history[f'val_loss'], label='val' + exp_name)
    # ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history[metric],
             label=f'train {metric}' + exp_name)
    ax2.plot(history.history[f'val_{metric}'],
             label=f'val {metric}' + exp_name)
    # ax2.set_ylim(0.25, 1.)
    ax2.set_title(f'{metric.capitalize()}')
    ax2.legend()
    return (ax1, ax2)


def get_train_test_datasets(
        dir_train_0=os.environ.get('DOGS_DATASET_TRAIN_PATH') + '/0',
        dir_train_1=os.environ.get('DOGS_DATASET_TRAIN_PATH') + '/1',
        dir_test_1=os.environ.get('DOGS_DATASET_TEST_PATH') + '/1',
        batch_size=602,
        img_width=int(int(os.environ.get('WIDTH'))),
        img_height=int(int(os.environ.get('HEIGHT')))):

    '''
    Get the train, validation and test datasets
    '''

    # labels = last split of the path
    label_dir_train_0 = dir_train_0.split('/')[-1]
    label_dir_train_1 = dir_train_1.split('/')[-1]
    label_dir_test_1 = dir_test_1.split('/')[-1]

    # generate train_tmp dir name from timestamp
    train_tmp_dir_name = dir_train_0 + '/../../' + \
        'Train_tmp_' + str(int(time.time()))

    # generate test_tmp dir name from timestamp
    test_tmp_dir_name = dir_test_1 + '/../../' + \
        'Test_tmp_' + str(int(time.time()))

    # create temp dir
    os.makedirs(train_tmp_dir_name, exist_ok=True)

    # create temp dir for train_0
    os.makedirs(train_tmp_dir_name + '/' + label_dir_train_0, exist_ok=True)

    # create temp dir for train_1
    os.makedirs(train_tmp_dir_name + '/' + label_dir_train_1, exist_ok=True)

    # copy content of dir_train_0 to tmp
    os.system('cp -r ' + dir_train_0 + '/* ' +
              train_tmp_dir_name + '/' + label_dir_train_0)

    # copy content of dir_train_1 to tmp
    os.system('cp -r ' + dir_train_1 + '/* ' +
              train_tmp_dir_name + '/' + label_dir_train_1)

    # create temp dir
    os.makedirs(test_tmp_dir_name, exist_ok=True)

    # create temp dir for test_1
    os.makedirs(test_tmp_dir_name + '/' + label_dir_test_1, exist_ok=True)

    # copy content of dir_test_1 to tmp
    os.system('cp -r ' + dir_test_1 + '/* ' +
              test_tmp_dir_name + '/' + label_dir_test_1)

    data_train = image_dataset_from_directory(
        train_tmp_dir_name,
        labels="inferred",
        label_mode="categorical",
        seed=123,
        image_size=(img_width, img_height),
        batch_size=batch_size,
    )

    data_test = image_dataset_from_directory(
        test_tmp_dir_name,
        labels="inferred",
        label_mode="categorical",
        seed=123,
        image_size=(img_width, img_height),
        batch_size=batch_size,
    )

    # Convert the TensorFlow dataset to a NumPy array to use train_test_split
    # Assuming datasets contains only one batch
    features_train, labels_train = next(iter(data_train))
    features_train = features_train.numpy()
    labels_train = labels_train.numpy()

    features_test, labels_test = next(iter(data_test))
    features_test = features_test.numpy()
    labels_test = labels_test.numpy()

    train_features, val_features, train_labels, val_labels = train_test_split(
        features_train, labels_train, test_size=0.2, random_state=42
    )
    # Create new TensorFlow datasets from the splits
    train_dataset = Dataset.from_tensor_slices((train_features, train_labels))
    val_dataset = Dataset.from_tensor_slices((val_features, val_labels))
    test_dataset = Dataset.from_tensor_slices((features_test, labels_test))

    # Optionally, you can further configure your TensorFlow datasets
    # For example, you can shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(
        buffer_size=10000).batch(batch_size=8)
    val_dataset = val_dataset.batch(batch_size=8)
    test_dataset = test_dataset.batch(batch_size=8)

    # remove tmp dir
    os.system('rm -rf ' + train_tmp_dir_name)
    os.system('rm -rf ' + test_tmp_dir_name)

    return train_dataset, val_dataset, test_dataset, features_train, features_test


'''
Get the dataset from the URL and unzip it
'''


def get_dataset():
    # URL of the zip file
    url = os.environ.get('DATASET_ZIP_PATH')

    # Get the dataset path from the environment variables
    dataset_path = './'

    # Create the directory if it does not exist
    os.makedirs(dataset_path, exist_ok=True)

    # Download the file
    output = os.path.join(dataset_path, 'file.zip')
    gdown.download(url, output, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)

    # remove file.zip
    os.remove(output)

    # remove dir __MACOSX and all its content
    os.system('rm -rf ' + dataset_path + '__MACOSX')

    # remove dir .DS_Store
    os.system('rm -rf ' + dataset_path + '.DS_Store')
