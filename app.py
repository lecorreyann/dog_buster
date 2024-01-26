import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from utils import *
from utils_gaspar import *
import pandas as pd
import numpy as np
from utils_yann import sign_in, sign_up, sign_out


st.set_page_config(page_title='DogBuster', page_icon='🐶',
                   layout='wide', initial_sidebar_state='auto')

st.session_state["showSignIn"] = True
st.session_state["showSignUp"] = False


# ------ SET UP DB ---------------
# Create tables
create_tables()

# ------ AUTHENTICATION ----------

query_params = st.experimental_get_query_params()
if st.session_state.get("authentication_status", False) == False:
    if 'sign_up' in query_params:
        sign_up()
    else:
        sign_in()


if st.session_state.get("authentication_status") == True:

    # ------ SIDEBAR ----------
    st.sidebar.title('DogBuster')
    # load image from imgs folder
    st.sidebar.image(
        'imgs/E814FBB9-2867-491D-A1F1-4B3370CC1DD6.png', width=100)
    st.sidebar.subheader(f'Welcome {st.session_state.get("name")}')
    # authenticator.logout("Logout", "sidebar")

    with st.sidebar:
        selected = option_menu(
            menu_title='DogBuster',
            options=['Home', 'Lost my Dog', 'Found a Dog'],
            icons=['house', 'tencent-qq', 'person-arms-up'],
            menu_icon='cast'
        )

        if st.button('Logout'):
            sign_out()

    # ------ MAIN ----------
    if selected == 'Home':
        st.write('Welcome to the Dog Buster App')
        st.write('This app is designed to help you find your lost dog')
        st.write('Please click on the tabs to navigate')
        lat, lon = get_lon_lat()
        if lat and lon:
            df = pd.DataFrame(
                np.random.randn(1000, 2) / [50, 50] + [lat, lon],
                columns=['lat', 'lon'])
            st.map(df)

    # ------ LOST MY DOG ----------
    if selected == 'Lost my Dog':
        uploaded_lost_dog_files = st.file_uploader("Choose some images...", type=[
                                                   "png", "jpg", "jpeg"], accept_multiple_files=True)
        if st.button('Upload lost animal'):
            for uploaded_file in uploaded_lost_dog_files:
                bytes_data = uploaded_file.read()
                st.write("filename:", uploaded_file.name)
                st.image(bytes_data, width=300)
                url = upload_image(bytes=bytes_data, user_id=st.session_state.get(
                    'user_id'), table_name='mascotas')

            # Generete the database
            train, val = get_dir_train_dir_val(user_id=st.session_state.get(
                'user_id'))

            # Upload model
            model = build_model()

            # Train model
            model, history, test_loss, test_acc = model_fit(model, dir_train=train,
                                                            dir_validation=val)

            # Upload model to DB
            model_name = f'models/model_{st.session_state.get("user_id")}.h5'
            tf.keras.models.save_model(model, model_name)
            remove_dir_train_dir_val(user_id=st.session_state.get('user_id'))
            upload_model(model_name)

    # ------ FOUND A DOG ----------
    if selected == 'Found a Dog':

        # load the model
        # model = tf.keras.models.load_model(f'models/model_{get_user_id(username)}.h5')

        # Give options to the user
        options = st.selectbox('Choose an option', [
                               'upload an image', 'take a photo'])

        # Gather the data with a camera
        if 'take a photo' in options:
            img_file = st.camera_input('Take a photo 📸')
            if st.button('Upload photo'):
                if img_file is not None:
                    bytes_data = img_file.getvalue()

                    # Upload the images
                    url = upload_image(bytes=bytes_data, user_id=st.session_state.get(
                        'user_id'), table_name='animales')
                    # predict with model
                    img = img_to_array_from_api(url)
                    find_owner(img=img, url=url)

        # Gather the data with a file uploader
        if 'upload an image' in options:
            img_file_buffer = st.file_uploader(
                "Upload an image", type=["png", "jpg", "jpeg"])
            if img_file_buffer is not None:
                st.image(img_file_buffer)
                bytes_data = img_file_buffer.read()

                # Upload the image
                url = upload_image(bytes=bytes_data, user_id=st.session_state.get(
                    'user_id'), table_name='animales')
                img = img_to_array_from_api(url)
                find_owner(img=img, url=url)
