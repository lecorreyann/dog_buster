# return VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from preprocess import balance_datasets
from utils import *
import os

# VGG16 model
def vgg16(input_shape):
    # input layer
    input_tensor = layers.Input(shape=input_shape)
    # VGG16 model
    model = VGG16(include_top=False, input_tensor=input_tensor)
    return model

def set_non_trainable(model:Sequential):
    # set non trainable layers
    model.trainable = False
    return model
# balance datasets


def add_last_layers(num_classes=2):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    model = vgg16(input_shape=(255,255,3))
    model = set_non_trainable(model)
    flattening_layer=layers.Flatten()
    dense_layer = layers.Dense(64,activation='relu')
    predict_layer = layers.Dense(num_classes,activation='softmax')

    model = Sequential([
        model,
        flattening_layer,
        dense_layer,
        predict_layer
    ])
    return model
model = add_last_layers()

print(model.summary())
