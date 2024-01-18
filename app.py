import hmac
import streamlit as st
import streamlit_geolocation as sg
import tensorflow as tf
from PIL import Image
from utils import *
from cloudinary import config, uploader, CloudinaryImage
import os
import requests
from utils_gaspar import *


config(
cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
api_key = os.environ.get('CLOUDINARY_CLOUD_KEY'),
api_secret = os.environ.get('CLOUDINARY_CLOUD_SECRET_KEY')
)
def check_password():
    '''Returns True if the password is correct'''
    def login_form():
        '''Form with widgets to collect user input'''
        with st.form(key='login'):
            username = st.text_input('Username', key='username')
            password = st.text_input('Password', key='password', type='password')
            submit_button = st.form_submit_button(label='Submit')
            return username, get_user_id(username)
    st.write('Do you have an account?')
    options = st.radio('',['Yes','No'])
    if options == 'Yes':
        return login_form()
    elif options == 'No':
        return create_user()

def create_user():
    '''Creates a user in the database'''
    username = st.text_input('Username', key='username')
    password = st.text_input('Password', key='password', type='password')
    if st.button('Submit'):
        if username and password:
            # Connect to the database
            upload_user(username,password)
            st.write('User created successfully')
            return username, get_user_id(username)
        else:
            st.error('Please fill in the form')
            return False
create_tables()
username,user_id =check_password()
st.write(user_id)
tab1,tab2,tab3 = st.tabs(['Home','Lost my Dog','Found a Dog'])
with tab1:
    st.write('Welcome to the Dog Buster App')
    st.write('This app is designed to help you find your lost dog')
    st.write('Please click on the tabs to navigate')
with tab2:
    uploaded_lost_dog_files = st.file_uploader("Choose some images...",type=["png", "jpg", "jpeg"],accept_multiple_files=True)
    if st.button('Upload lost animal'):
        for uploaded_file in uploaded_lost_dog_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            st.image(bytes_data, use_column_width=True)
            st.write(type(bytes_data))
            result = upload_img_to_cloudinary(bytes_data, public_id = f"lost_dog_{time.time()}")
            img = img_to_array_from_api(result['url'])
            upload_user_pet(username,result['url'])
with tab3:

    #load the model
    model = tf.keras.models.load_model('models/model_Resnet_1.h5')

    # Give options to the user
    options = st.selectbox('Choose an option',['upload an image','take a photo'])

    # Get the location of the user
    lat,lon = get_lon_lat()
    # Gather the data with a camera
    if 'take a photo' in options:
        img_file = st.camera_input('Take a photo 📸')
        if img_file is not None:
            bytes_data = img_file.getvalue()

            # Upload the image to cloudinary
            result = upload_img_to_cloudinary(bytes_data, public_id = f"dog_found_{time.time()}")

            # Fetch the image from cloudinary
            img = img_to_array_from_api(result['url'])

            # Upload the image to the database
            upload_animal(result['url'],lat,lon)
            #predict with model
            prediction = model.predict(img)
            st.write(prediction)

    # Gather the data with a file uploader
    if 'upload an image' in options:
        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            bytes_data = Image.open(img_file_buffer)

            # Upload the image to cloudinary
            result = upload_img_to_cloudinary(bytes_data, public_id = f"dog_found_{time.time()}")

            #predict with model
            img = img_to_array_from_api(result['url'])

            # Upload the image to the database
            upload_animal(result['url'],lat,lon)
            prediction = model.predict(img)
            st.write(prediction)
