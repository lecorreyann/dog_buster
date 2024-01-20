import cloudinary
from cloudinary.uploader import upload
import sqlite3
import requests
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import imgaug.augmenters as iaa
from utils_gaspar import create_tables,get_address

cloudinary.config(
cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
api_key = os.environ.get('CLOUDINARY_CLOUD_KEY'),
api_secret = os.environ.get('CLOUDINARY_CLOUD_SECRET_KEY')
)

def upload_image(bytes, user_id, table_name, lat=None, ln=None, address=None, found=False):
    # Upload image to cloudinary
    response = upload(bytes, folder="dog_buster")
    # Get the url
    url = response["secure_url"]
    # Connect to the database
    cursor, connection = create_tables()
    # Insert the image url into the database
    if (table_name == "animales"):
        address,lat,ln = get_address()
        address = f"""{address['address'].get('road','')} {address['address'].get('house_number','')},
            {address['address'].get('town','')}, {address['address'].get('country','')}"""
        insert_image = '''
            INSERT INTO animales (url,lat,ln,address,found)
            VALUES (?, ?, ?, ?, ?)
        '''
        # Execute the query
        cursor.execute(insert_image, (url, lat, ln, address, found))
    elif (table_name == "mascotas"):
        insert_image = '''
            INSERT INTO mascotas (url, user_id)
            VALUES (?, ?)
        '''
        # Execute the query
        cursor.execute(insert_image, (url, user_id))
    # Commit the changes
    connection.commit()
    # Close the connection
    connection.close()
    return url

def download_image(url, dir):
    # Download image from the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the file name from the URL
        file_name = os.path.join(dir, url.split("/")[-1])

        # Save the image content to a local file
        with open(file_name, 'wb') as file:
            file.write(response.content)


def augmentate_picture(url, output_dir, num_augmentations):
    # Transform the image into an array
    img = load_img(url)
    img_array = img_to_array(img)
    # Reshape the array
    img_array = img_array.reshape((1,) + img_array.shape)
    # Create a generator
    datagen = ImageDataGenerator(
        zoom_range=0.2,       # Randomly zoom in/out
        horizontal_flip=True,  # Randomly flip horizontally
        vertical_flip=True,    # Randomly flip vertically
        rotation_range=45      # Randomly rotate the image up to 45 degrees
    )
    # Generate num_augmentations augmented images
    i = 0
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix="augmented", save_format="jpeg"):
        i += 1
        if i >= num_augmentations:
            break


wished_train_qt = 50  # Number of images wished to train the model
wished_val_qt = 15  # Number of images wished to validate the model
# Number of images wished to train and validate the model
wished_images_qt = wished_train_qt + wished_val_qt


def get_dir_train_dir_val(user_id):
    """
    DIR TRAIN
    """
    # Connect to the database
    cursor, connection = create_tables()
    # Get the url
    cursor.execute(f"SELECT url FROM mascotas WHERE user_id = {user_id}")
    # Get the results
    results = cursor.fetchall()
    # Close the connection
    connection.close()
    # Get only the urls
    urls_mascotas = [url[0] for url in results]

    # Create a temporary train directory
    if not os.path.exists("train_" + str(user_id)):
        os.mkdir("train_" + str(user_id))

    # Create a temporary directory inside the train directory to store the images user uploaded
    if not os.path.exists("train_" + str(user_id) + "/target"):
        os.mkdir("train_" + str(user_id) + "/target")

    # Download the images from the urls inside the train/target directory
    for url in urls_mascotas:
        download_image(url, "train_" + str(user_id) + "/target")

    # Count the number of images inside the train_<user_id>/target directory
    image_train_qt = len([file for file in os.listdir("train_" + str(user_id) + "/target")
                          if os.path.isfile(os.path.join("train_" + str(user_id) + "/target", file))])

    # If the number of images is less than the wished number of images to train the model
    if image_train_qt < wished_images_qt:
        # Get all the images path inside the train_<user_id>/target directory
        images_path = [os.path.join("train_" + str(user_id) + "/target", file) for file in
                       os.listdir("train_" + str(user_id) + "/target")]

        # While the number of images is less than the wished number of images to train the model
        while image_train_qt < wished_images_qt:
            # Loop through the images path
            for image_path in images_path:
                # If the number of images is equal to the wished number of images to train the model
                if image_train_qt == wished_images_qt:
                    break

                # Augmentate the images
                augmentate_picture(
                    url=image_path, output_dir="train_" + str(user_id) + "/target", num_augmentations=1)
                # Count the number of images inside the train/target directory
                image_train_qt = len([file for file in os.listdir("train_" + str(user_id) + "/target")
                                      if os.path.isfile(os.path.join("train_" + str(user_id) + "/target", file))])
                print(image_path)
                print(image_train_qt)
    # Create a temporary directory inside the train directory to store the images of other users
    if not os.path.exists("train_" + str(user_id) + "/other"):
        os.mkdir("train_" + str(user_id) + "/other")

    # Copy/Past the images from ./Dataset/Train to ./train_<user_id>/other
    for file in os.listdir("Dataset/Train"):
        os.system(f"cp Dataset/Train/{file} train_{user_id}/other")

    """
    DIR VAL
    """
    # Create a temporary val directory
    if not os.path.exists("val_" + str(user_id)):
        os.mkdir("val_" + str(user_id))

    # Create a temporary directory inside the val directory to store the images user uploaded
    if not os.path.exists("val_" + str(user_id) + "/target"):
        os.mkdir("val_" + str(user_id) + "/target")

    # Cut and Past 15 random images from train_<user_id>/target to val_<user_id>/target
    for file in os.listdir("train_" + str(user_id) + "/target"):
        os.system(f"mv train_{user_id}/target/{file} val_{user_id}/target")
        if len(os.listdir("val_" + str(user_id) + "/target")) == wished_val_qt:
            break

    # Create a temporary directory inside the val directory to store the images of other users
    if not os.path.exists("val_" + str(user_id) + "/other"):
        os.mkdir("val_" + str(user_id) + "/other")

    # Copy/Past the images from ./Dataset/Validation to ./val_<user_id>/other
    for file in os.listdir("Dataset/Validation"):
        os.system(f"cp Dataset/Validation/{file} val_{user_id}/other")

    # Return name of the train and val directories
    return "train_" + str(user_id), "val_" + str(user_id)


def remove_dir_train_dir_val(user_id):
    # Empty the train/target directory
    for file in os.listdir("train_" + str(user_id) + "/target"):
        os.remove("train_" + str(user_id) + "/target/" + file)
    # Delete the train/target directory
    os.rmdir("train_" + str(user_id) + "/target")

    # Empty the train/other directory
    for file in os.listdir("train_" + str(user_id) + "/other"):
        os.remove("train_" + str(user_id) + "/other/" + file)
    # Delete the train/other directory
    os.rmdir("train_" + str(user_id) + "/other")

    # Delete the train directory
    os.rmdir("train_" + str(user_id))

    # Empty the val/target directory
    for file in os.listdir("val_" + str(user_id) + "/target"):
        os.remove("val_" + str(user_id) + "/target/" + file)
    # Delete the val/target directory
    os.rmdir("val_" + str(user_id) + "/target")

    # Empty the val/other directory
    for file in os.listdir("val_" + str(user_id) + "/other"):
        os.remove("val_" + str(user_id) + "/other/" + file)
    # Delete the val/other directory
    os.rmdir("val_" + str(user_id) + "/other")

    # Delete the val directory
    os.rmdir("val_" + str(user_id))


learning_rate = 0.0001
l2_value = 0.0001


def build_model(
        learning_rate=learning_rate,
        l2_value=l2_value):
    # Load VGG16
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    # Freeze the layers
    base_model.trainable = False
    # Create the model architecture
    vgg16 = Sequential()
    vgg16.add(base_model)
    vgg16.add(Flatten())
    vgg16.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_value)))
    vgg16.add(Dropout(0.3))
    vgg16.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_value)))
    vgg16.add(Dense(1, activation='sigmoid'))
    # Compile the model
    vgg16.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return vgg16


batch_size = 32
epochs = 20
patience = 5


def model_fit(model, dir_train, dir_validation, batch_size=batch_size, epochs=epochs, patience=patience):

    # Data augmentation
    augmentation_seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Affine(rotate=(-45, 45)),
        iaa.SomeOf((0, 2), [
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.03, 0.15), size_percent=(
                0.02, 0.05), per_channel=0.2),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.ContrastNormalization(
                (0.5, 2.0), per_channel=0.5)),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.Grayscale(alpha=(0.0, 1.0)),
        ]),
    ], random_order=True)

    # Create the generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        dir_validation,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Fit the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(validation_generator)
    # test_acc is the accuracy on the validation set
    # test_loss is the loss on the validation set

    return model, history, test_loss, test_acc
