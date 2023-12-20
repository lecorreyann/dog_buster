# return VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
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
    dense_layer_1 = layers.Dense(64,activation='relu')
    dense_layer_2 = layers.Dense(128,activation='relu')

    predict_layer = layers.Dense(num_classes,activation='softmax')

    model = Sequential([
        model,
        flattening_layer,
        dense_layer_1,
        dense_layer_2,
        predict_layer
    ])
    return model
def build_model(num_classes=2):
    # $CHALLENGIFY_BEGIN
    model = add_last_layers(num_classes=num_classes)

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
