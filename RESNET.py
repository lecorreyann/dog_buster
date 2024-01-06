from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def build_resnet(num_classes=2):
    resnet_model = Sequential()
    pretrained_model = ResNet50(include_top=False, input_shape=(
        255, 255, 3), pooling='maxpooling', weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False
    resnet_model.add(pretrained_model)
# Add fully connected layers for classification
    resnet_model.add(layers.Flatten())
    resnet_model.add(layers.Dense(124, activation='relu'))
    resnet_model.add(layers.Dropout(0.45))
    resnet_model.add(layers.Dense(54, activation='relu'))
    resnet_model.add(layers.Dense(num_classes, activation='softmax'))
# Compile the model
    resnet_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return resnet_model
