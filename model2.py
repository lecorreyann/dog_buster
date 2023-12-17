from tensorflow.keras import models, layers
from tensorflow.keras.metrics import Recall
# alternativa de modelos : >>>> VGG16, restnet, google vision


def init():
    input_shape = (255, 255, 3)
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(5, 5), input_shape=input_shape,
                      padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=(5, 5),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(4, 4)),
        layers.Conv2D(128, kernel_size=(4, 4),
                      padding='same', activation='relu'),
        layers.Conv2D(160, kernel_size=(4, 4),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3)),
        layers.Conv2D(256, kernel_size=(3, 3),
                      padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(30, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    return model


def compile(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=[Recall()])
    return model
