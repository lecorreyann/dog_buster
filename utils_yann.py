import streamlit as st
from utils_gaspar import create_tables
import hashlib


# Function to verify user credentials
def verify_credentials(email, provided_password):
    cursor, connection = create_tables()

    # Retrieve the stored password
    cursor.execute('SELECT * FROM usuarios WHERE email = ?', (email,))
    user = cursor.fetchone()
    connection.close()

    print(user)

    if user is None:
        return False
    else:
        user = {'id': user[3], 'name': user[0],
                'email': user[1], 'password': user[2]}

        print(user)
        # Hash the provided password
        provided_password_hash = hashlib.sha256(
            provided_password.encode()).hexdigest()

        # Compare the provided password hash with the stored password hash
        if provided_password_hash == user.get('password'):
            return user
        return False


def sign_in():
    # Sign-in form
    with st.form(key='sign_in_form'):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label='Sign In')

        if submit_button:
            user = verify_credentials(email, password)
            if user:
                st.success('Logged in successfully')
                st.session_state["authentication_status"] = True
                st.session_state["name"] = user.get('name')
                st.session_state["email"] = user.get('email')
                st.session_state["user_id"] = user.get('id')
            else:
                st.error('Email/password is incorrect')
    # Display notice xmessage 'You don't have an account? Sign up'
    st.markdown('You don\'t have an account? <a href="?sign_up" target="_self">Sign up</a>',
                unsafe_allow_html=True)


def sign_up():
    # Sign-up form
    with st.form(key='sign_up_form'):
        new_name = st.text_input("Name")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label='Sign Up')

        if submit_button:
            cursor, connection = create_tables()
            cursor.execute('SELECT * FROM usuarios WHERE email = ?',
                           (new_email,))
            data = cursor.fetchone()
            if data is None:
                # hash the password
                hashed_password = hashlib.sha256(
                    new_password.encode()).hexdigest()
                cursor.execute('INSERT INTO usuarios (name, email, password) VALUES (?, ?,?)',
                               (new_name, new_email, hashed_password))
                connection.commit()
                st.success('You have successfully created a new account')
                st.info('Login to your new account')
            else:
                st.warning('Email already exists')
            connection.close()
    st.markdown('You already have an account? <a href="?sign_in" target="_self">Sign in</a>',
                unsafe_allow_html=True)


def sign_out():
    st.session_state["authentication_status"] = False
    st.experimental_rerun()
