from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from datetime import datetime
import os
import numpy as np


def data_transformation(img_array, output_dir, qty=1):
    for i in range(qty):
        datagen = ImageDataGenerator(
            # Random rotation in the range [-30, 30] degrees
            rotation_range=np.random.randint(-30, 30),
            # zoom_range=[0.8, 1.2],  # Random zoom in the range [0.8, 1.2]
            # width_shift_range=0.1,  # Random horizontal shift
            # height_shift_range=0.1,  # Random vertical shift
            # shear_range=0.2,  # Random shear
            # brightness_range=[0.5, 1.5],  # Random brightness adjustment
            # channel_shift_range=20,  # Random channel shift
            # horizontal_flip=True,  # Random horizontal flip
            # vertical_flip=True,  # Random vertical flip
            # fill_mode='nearest',  # Fill mode for handling newly created pixels
            # Custom preprocessing function (random blur)
            # preprocessing_function=lambda x: x * np.random.uniform(0.9, 1.1)
        )

        # Generate one augmented image
        augmented_image = next(datagen.flow(img_array, batch_size=1))[
            0].astype(np.uint8)

        # Specify the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save the augmented image to the output directory
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H%M%S")
        output_path = os.path.join(
            output_dir, f"augmented_image_{formatted_datetime}.jpg")
        array_to_img(augmented_image).save(output_path)
