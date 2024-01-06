# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.preprocessing import image
from datetime import datetime
import numpy as np
import cv2


def data_transformation(img_array, output_dir, qty=1):
    print(output_dir)
    for i in range(qty):

        # get image from array
        img = img_array[0]

        # flip image from array with cv2
        # 0: flip vertically, 1: flip horizontally, -1: flip both
        img = cv2.flip(img, np.random.randint(-1, 2))

        # zoom image from array with cv2
        # random zoom in the range [0.8, 1.2]
        zoom_factor = np.random.uniform(0.8, 1.2)
        # get image size
        height, width, channels = img.shape
        # zoom image
        img = cv2.resize(img, (int(width * zoom_factor),
                         int(height * zoom_factor)))

        # get current date and time
        now = datetime.now()
        # convert to string
        dt_string = now.strftime("%d%m%Y%H%M%S")
        # filename
        filename = output_dir + '/augmented_' + dt_string + '.jpg'

        # save image
        cv2.imwrite(filename, img)

        print(f"Saved image to {filename}")
