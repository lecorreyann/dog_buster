from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, Input, Lambda
import os


def build_encoder(latent_dimension, width=int(os.environ.get('WIDTH')), height=int(os.environ.get('HEIGHT')), channels=3):
    '''returns an encoder model, of output_shape equals to latent_dimension'''
    encoder = Sequential()

    encoder.add(Conv2D(8, (2, 2), input_shape=(
        width, height, channels), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='tanh'))

    return encoder


def build_decoder(latent_dimension):
    # $CHALLENGIFY_BEGIN
    decoder = Sequential()

    decoder.add(Dense(75*75*8, activation='tanh',
                input_shape=(latent_dimension,)))
    decoder.add(Reshape((75, 75, 8)))  # no batch axis here
    decoder.add(Conv2DTranspose(8, (2, 2), strides=2,
                padding='same', activation='relu'))

    decoder.add(Conv2DTranspose(1, (2, 2), strides=2,
                padding='same', activation='relu'))  # no batch axis here
    # divide by 4 because of the 2 strides

    return decoder
    # $CHALLENGIFY_END


def build_autoencoder(encoder, decoder, width=int(os.environ.get('WIDTH')), height=int(os.environ.get('HEIGHT')), channels=3):
    inp = Input((width, height, channels))
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder


def compile_autoencoder(autoencoder):
    # $CHALLENGIFY_BEGIN
    autoencoder.compile(loss='mse',
                        optimizer='adam')
