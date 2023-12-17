from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
# alternativa de modelos : >>>> VGG16, restnet, google vision


def init():
    model = models.Sequential([
        layers.Conv2D(16, kernel_size=(6, 6), input_shape=(
            255, 255, 3), padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3)),
        layers.Flatten(),
        layers.Dense(6, activation='relu'),
        layers.Dense(2, activation='softmax')

    ])
    return model


def compile(model):
    optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=[Recall()])
    return model
