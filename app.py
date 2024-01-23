import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import tensorflow as tf
from utils import *
from utils_gaspar import *

create_tables()
users = get_all_users()
emails=[]
names=[]
usernames=[]
passwords=[]
for user in users:
    emails.append(user[2])
    names.append(user[1])
    usernames.append(user[0])
    passwords.append(user[3])

credentials = {'usernames':{}}

for index in range(len(emails)):
    credentials['usernames'][usernames[index]] = {'name':names[index],
                                                  'email':emails[index],
                                                  'password':passwords[index]}

authenticator = stauth.Authenticate(credentials,'DogBuster','abcdef')
name,authentification_satus,username = authenticator.login("Login","main")
if not authentification_satus:
    sing_up()
if authentification_satus == False:
    st.error('Username/password is incorrect')
if authentification_satus == None:
    st.write('Please enter your username and password')

if authentification_satus:

    # ------ SIDEBAR ----------
    st.sidebar.subheader(f'Welcome {name}')
    authenticator.logout("Logout","sidebar")
    with st.sidebar:
        selected = option_menu(
            menu_title='DogBuster',
            options = ['Home','Lost my Dog','Found a Dog'],
            icons=['house','tencent-qq','person-arms-up'],
            menu_icon='cast'
        )

    # ------ MAIN ----------
    if selected == 'Home':
        st.write('Welcome to the Dog Buster App')
        st.write('This app is designed to help you find your lost dog')
        st.write('Please click on the tabs to navigate')
        st.write(get_user_email(get_user_id(username)))
        st.write(get_user_name(get_user_id(username)))

    # ------ LOST MY DOG ----------
    if selected == 'Lost my Dog':
        uploaded_lost_dog_files = st.file_uploader("Choose some images...",type=["png", "jpg", "jpeg"],accept_multiple_files=True)
        if st.button('Upload lost animal'):
            for uploaded_file in uploaded_lost_dog_files:
                bytes_data = uploaded_file.read()
                st.write("filename:", uploaded_file.name)
                st.image(bytes_data, use_column_width=True)
                url = upload_image(bytes=bytes_data,user_id=get_user_id(username),table_name='mascotas')

            # Generete the database
            train,val=get_dir_train_dir_val(user_id=get_user_id(username))

                # Upload model
            model = build_model()

                # Train model
            model,history, test_loss, test_acc = model_fit(model,dir_train=train,
                                                               dir_validation=val)

                # Upload model to DB
            model_name = f'models/model_{get_user_id(username)}.h5'
            tf.keras.models.save_model(model,model_name)
            remove_dir_train_dir_val(user_id=get_user_id(username))
            upload_model(username,model_name)


    # ------ FOUND A DOG ----------
    if selected == 'Found a Dog':

        #load the model
        #model = tf.keras.models.load_model(f'models/model_{get_user_id(username)}.h5')

        # Give options to the user
        options = st.selectbox('Choose an option',['upload an image','take a photo'])

        # Gather the data with a camera
        if 'take a photo' in options:
            img_file = st.camera_input('Take a photo 📸')
            if st.button('Upload photo'):
                if img_file is not None:
                    bytes_data = img_file.getvalue()

                    # Upload the images
                    url = upload_image(bytes=bytes_data,user_id=get_user_id(username),table_name='animales')

                    #predict with model
                    img = img_to_array_from_api(url)
                #prediction = model.predict(img)
                #st.write(prediction)

        # Gather the data with a file uploader
        if 'upload an image' in options:
            img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if img_file_buffer is not None:
                st.image(img_file_buffer)
                bytes_data = img_file_buffer.read()

                # Upload the image
                url = upload_image(bytes=bytes_data,user_id=get_user_id(username),table_name='animales')
                img = img_to_array_from_api(url)
                #prediction = model.predict(img)
                #st.write(prediction)
                #if prediction[0, 0] >= 0.5:
                #   st.success('This is your dog')
                send_email(url=url,user_id=get_user_id(username))
