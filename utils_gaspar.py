import sqlite3
import requests
import os
from cloudinary import CloudinaryImage, uploader, config
from io import BytesIO
from PIL import Image
from utils import *
import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from streamlit_js_eval import streamlit_js_eval, copy_to_clipboard, create_share_link, get_geolocation

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
            address VARCHAR(255)
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
            email VARCHAR(255) UNIQUE,
            password VARCHAR(255),
            id INTEGER PRIMARY KEY AUTOINCREMENT
        );
    '''
    cursor.execute(create_user_tables)
    cursor.execute(create_animals_tables)
    cursor.execute(create_pet_tables)
    connection.commit()
    return cursor,connection

def upload_user(username,password):
    '''Uploads the user to the database'''
    # Connect to the database
    cursor,connection = create_tables()
    # Insert the data
    insert_query = f'''
        INSERT INTO usuarios (email,password)
        VALUES ('{username}','{password}')
    '''
    cursor.execute(insert_query)
    connection.commit()
    # Close the connection
    connection.close()
    return True

def get_if_password_correct(username,password):
    '''Get the user id'''
    # Connect to the database
    cursor,connection = create_tables()
    # Insert the data
    insert_query = f'''
        SELECT password FROM usuarios WHERE email = '{username}'
    '''
    cursor.execute(insert_query)
    password_db = cursor.fetchone()[0]
    connection.commit()
    # Close the connection
    connection.close()
    if password_db == password:
        return True
    else:
        return False

def get_user_id(username):
    '''Get the user id'''
    # Connect to the database
    cursor,connection = create_tables()
    # Insert the data
    insert_query = f'''
        SELECT id FROM usuarios WHERE email = '{username}'
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
