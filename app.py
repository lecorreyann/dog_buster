import streamlit as st
import tensorflow as tf
from PIL import Image
from utils import *
from cloudinary import config, uploader, CloudinaryImage
import os
import requests

config(
cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
api_key = os.environ.get('CLOUDINARY_CLOUD_KEY'),
api_secret = os.environ.get('CLOUDINARY_CLOUD_SECRET_KEY')
)
tab1,tab2,tab3 = st.tabs(['Home','Lost my Dog','Found a Dog'])
with tab1:
    st.write('Welcome to the Dog Buster App')
    st.write('This app is designed to help you find your lost dog')
    st.write('Please click on the tabs to navigate')
with tab2:
    uploaded_lost_dog_files = st.file_uploader("Choose some images...",type=["png", "jpg", "jpeg"],accept_multiple_files=True)
    if st.button('Submit'):
        for uploaded_file in uploaded_lost_dog_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            st.image(bytes_data, use_column_width=True)
            st.write(type(bytes_data))
        #load the model
        model = tf.keras.models.load_model('models/model_Resnet_1.h5')
        #train the model
        #model.fit()
        #save the model
with tab3:
    #load the model
    model = tf.keras.models.load_model('models/model_Resnet_1.h5')
    options = st.selectbox('Choose an option',['upload an image','take a photo'])
    if 'take a photo' in options:
        img_file = st.camera_input('Take a photo 📸')
        if img_file is not None:
            bytes_data = img_file.getvalue()
            result = upload_img_to_cloudinary(bytes_data, public_id = f"lost_dog_{time.time()}")
            img = img_to_array_from_api(result['url'])
            #predict with model
            prediction = model.predict(img)
            #st.write(prediction)
    if 'upload an image' in options:
        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            bytes_data = Image.open(img_file_buffer)
            result = upload_img_to_cloudinary(bytes_data, public_id = f"lost_dog_{time.time()}")
            #predict with model
            img = img_to_array_from_api(result['url'])
            prediction = model.predict(img)
            st.write(prediction)
