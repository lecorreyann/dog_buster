import sqlite3
import requests
import os
from cloudinary import CloudinaryImage, uploader, config
from io import BytesIO
from PIL import Image
from utils import *
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_geolocation import streamlit_geolocation
from streamlit_js_eval import streamlit_js_eval, copy_to_clipboard, create_share_link, get_geolocation
import re
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array_keras
import numpy as np

def create_tables():
    # Connect to the database
    connection = sqlite3.connect('animales.db')

    # Create a cursor
    cursor = connection.cursor()

    # Create a table if it doesn't exist
    create_animals_tables = '''
        CREATE TABLE IF NOT EXISTS animales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url VARCHAR(255),
            lat REAL,
            ln  REAL,
            address VARCHAR(255),
            found BOOLEAN DEFAULT FALSE
        );
        '''
    create_pet_tables ='''
        CREATE TABLE IF NOT EXISTS mascotas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url VARCHAR(255),
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES usuarios (id)
        );
        '''
    create_user_tables ='''
        CREATE TABLE IF NOT EXISTS usuarios (
            username VARCHAR(255) UNIQUE,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE,
            password VARCHAR(255),
            id INTEGER PRIMARY KEY AUTOINCREMENT
        );
    '''
    create_model_tables = '''
        CREATE TABLE IF NOT EXISTS modelos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) UNIQUE,
            user_id INTEGER
        );
    '''
    cursor.execute(create_user_tables)
    cursor.execute(create_animals_tables)
    cursor.execute(create_pet_tables)
    cursor.execute(create_model_tables)
    connection.commit()
    return cursor,connection

def get_all_users():
    '''Get all the users'''
    # Connect to the database
    cursor,connection = create_tables()
    # Insert the data
    insert_query = f'''
        SELECT * FROM usuarios
    '''
    cursor.execute(insert_query)
    users = cursor.fetchall()
    connection.commit()
    # Close the connection
    connection.close()
    return users

def upload_user(email,name,username,password):
    '''Uploads the user to the database'''
    # Connect to the database
    cursor,connection = create_tables()
    # Insert the data
    username = username.lower()
    insert_query = f'''
        INSERT INTO usuarios (username,name,email,password)
        VALUES ('{username}','{name}','{email}','{password}')
    '''
    cursor.execute(insert_query)
    connection.commit()
    # Close the connection
    connection.close()
    return True

def resize_img(img, height=int(os.environ.get('HEIGHT')), width=int(os.environ.get('WIDTH')), channels=3):
    '''
    Return a resized image
    '''

    # Resize the image
    img = np.resize(img, (height, width, channels))
    return img

def reshape_img(img, height=int(os.environ.get('HEIGHT')), width=int(os.environ.get('WIDTH')), channels=3):
    '''
    Return a reshaped image
    '''
    img = img.reshape((-1, height, width, channels))
    return img

def upload_model(username,model_name):
    '''Uploads the user to the database'''
    # Connect to the database
    cursor,connection = create_tables()
    # Get user_id
    user_id = get_user_id(username)
    # Insert the data
    insert_query = f'''
        INSERT INTO modelos (name,user_id)
        VALUES ('{model_name}','{user_id}')
    '''
    cursor.execute(insert_query)
    connection.commit()
    # Close the connection
    connection.close()
    return True

def reshape_img(img, height=int(os.environ.get('HEIGHT')), width=int(os.environ.get('WIDTH')), channels=3):
    '''
    Return a reshaped image
    '''
    img = img.reshape((-1, height, width, channels))
    return img

def resize_img(img, height=int(os.environ.get('HEIGHT')), width=int(os.environ.get('WIDTH')), channels=3):
    '''
    Return a resized image
    '''

    # Resize the image
    img = np.resize(img, (height, width, channels))
    return img

def check_model(username):
    try:
        '''Check if the user has a model'''
        # Connect to the database
        cursor,connection = create_tables()
        # Get user_id
        user_id = get_user_id(username)
        # Insert the data
        insert_query = f'''
            SELECT name FROM modelos WHERE user_id = '{user_id}'
        '''
        cursor.execute(insert_query)
        model_name = cursor.fetchone()[0]
        connection.commit()
        # Close the connection
        connection.close()
        return True
    except:
        return False

def get_user_id(username):
    '''Get the user id'''
    # Connect to the database
    cursor,connection = create_tables()
    # Insert the data
    insert_query = f'''
        SELECT id FROM usuarios WHERE username = '{username}'
    '''
    cursor.execute(insert_query)
    user_id = cursor.fetchone()[0]
    connection.commit()
    # Close the connection
    connection.close()
    return user_id

def upload_user_pet(username,url):
    '''Uploads the user to the database'''
    # Connect to the database
    cursor,connection = create_tables()
    # Get user_id
    user_id = get_user_id(username)
    # Insert the data
    insert_query = f'''
        INSERT INTO mascotas (url,user_id)
        VALUES ('{url}','{user_id}')
    '''
    cursor.execute(insert_query)
    connection.commit()
    # Close the connection
    connection.close()
    return True

def upload_animal(url):
    '''Uploads the user to the database'''
    # Connect to the database
    cursor,connection = create_tables()

    # Get the address
    address,lat,ln = get_address()
    st.write(f"""{address['address'].get('road','')} {address['address'].get('house_number','')},
            {address['address'].get('town','')}, {address['address'].get('country','')}""")

    address = f"""{address['address'].get('road','')} {address['address'].get('house_number','')},
            {address['address'].get('town','')}, {address['address'].get('country','')}"""
    # Insert the data

    insert_query = f'''
        INSERT INTO animales (url,lat,ln,address)
        VALUES ('{url}','{lat}','{ln}','{address}')
    '''
    cursor.execute(insert_query)
    connection.commit()
    #Close the connection
    connection.close()
    return True

def upload_img_to_cloudinary(bytes_data, public_id):
    '''
    Upload an image to cloudinary
    '''
    # upload image to cloudinary
    result=uploader.upload(bytes_data, public_id = public_id)
    return result

def getImage(url):
    '''
    Get an image from an url
    '''
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def img_to_array_from_api(url):
    '''
    Return a numpy array of an image
    '''
    # get image from url
    img = getImage(url)
    img = img_to_array_keras(img)
    img = np.expand_dims(img, axis=0)
    img = resize_img(img)
    img = reshape_img(img)
    # display the image
    # quit extra dimension
    # plt.imshow(array_to_img(img[0].astype(np.uint8)))
    # plt.show()
    return img

def get_lon_lat():
    '''
    get the user latitude and longitude
    '''
    loc = get_geolocation()
    if loc and 'coords' in loc:
        latitude, longitude = loc['coords']['latitude'], loc['coords']['longitude']
        print(f'Got user location:, {latitude}, {longitude}')
        return latitude,longitude
    else:
        print('No location found')
        return None

def get_address():
    '''
    get the user address
    '''
    latitude,longitude = get_lon_lat()
    location = requests.get(
            f'https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={latitude}&lon={longitude}').json()
    return location, latitude, longitude

def sing_up():
    with st.form(key='sing up', clear_on_submit=True):
        st.subheader(':green[Sign] up')
        email = st.text_input('Email', placeholder='Enter your Email')
        name = st.text_input('Name', placeholder='Enter your Name')
        username = st.text_input('Username', placeholder='Enter your Username')
        password = st.text_input('Password', type='password', placeholder='Enter your Password')
        password2 = st.text_input('Password', type='password', placeholder='Repeat your Password')
        if len(name) >=3:
            if len(password) >=5:
                if password == password2:
                    st.success('User created')
                    hashed_password = stauth.Hasher([password]).generate()
                    upload_user(email,name,username,hashed_password[0])
                else:
                    st.error('Passwords do not match')
            else:
                st.error('Password must be at least 5 characters')
        else:
            st.error('Name must be at least 3 characters')
        st.form_submit_button('Sign up')
