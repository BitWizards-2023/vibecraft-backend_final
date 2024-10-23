import logging
import requests
import firebase_admin
from firebase_admin import credentials, firestore, auth
from fastapi import HTTPException

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_spotify_user_info(spotify_token: str):
    """
    Retrieve Spotify user information using the access token.
    """
    headers = {
        'Authorization': f'Bearer {spotify_token}'
    }
    response = requests.get('https://api.spotify.com/v1/me', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(
            f"Failed to get Spotify user info: {response.status_code}")
        return None


def generate_firebase_token(spotify_token: str, additional_data: dict):
    """
    Generate a Firebase custom token using Spotify user information.
    Add additional custom data such as age, address, and gender to the token.
    """
    spotify_user_info = get_spotify_user_info(spotify_token)

    if spotify_user_info:
        spotify_user_id = spotify_user_info['id']
        spotify_email = spotify_user_info.get('email', '')

        # Retrieve additional fields from the request (from the Flutter app)
        user_age = additional_data.get('age', '')
        user_address = additional_data.get('address', '')
        user_gender = additional_data.get('gender', '')

        # Validate additional data (add more validation as needed)
        if not all([user_age, user_address, user_gender]):
            raise HTTPException(
                status_code=400, detail="Missing required user data")

        # Create Firebase UID from Spotify ID
        firebase_uid = f'spotify:{spotify_user_id}'

        try:
            # Check if user exists in Firebase
            try:
                user = auth.get_user(firebase_uid)
                user_exists = True
                logger.info(f"User {firebase_uid} exists in Firebase.")
            except firebase_admin.auth.UserNotFoundError:
                user_exists = False
                logger.info(
                    f"User {firebase_uid} does not exist. Creating new user.")

            if user_exists:
                # User exists, generate a token with additional claims
                additional_claims = {
                    'spotify_email': spotify_email,
                    'spotify_user_id': spotify_user_id,
                    'age': user_age,
                    'address': user_address,
                    'gender': user_gender
                }
                custom_token = auth.create_custom_token(
                    firebase_uid, additional_claims)
                return {
                    'firebase_token': custom_token.decode('utf-8'),
                    'is_new_user': False  # Return False for existing users
                }

            else:
                # Create new Firebase user if it doesn't exist
                user = auth.create_user(
                    uid=firebase_uid,
                    email=spotify_email,
                    email_verified=False,  # Set to True if email is verified
                    display_name=spotify_user_info.get('display_name', '')
                )

                # Save new user data to Firestore
                user_data = {
                    'display_name': spotify_user_info.get('display_name', ''),
                    'email': spotify_email,
                    'spotify_user_id': spotify_user_id,
                    'age': user_age,
                    'address': user_address,
                    'gender': user_gender,
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                save_user_to_firestore(firebase_uid, user_data)

                # Generate a custom token with additional claims for the new user
                additional_claims = {
                    'spotify_email': spotify_email,
                    'spotify_user_id': spotify_user_id,
                    'age': user_age,
                    'address': user_address,
                    'gender': user_gender
                }
                custom_token = auth.create_custom_token(
                    firebase_uid, additional_claims)
                return {
                    'firebase_token': custom_token.decode('utf-8'),
                    'is_new_user': True  # Return True for new users
                }

        except Exception as e:
            logger.error(f"Error generating Firebase token: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error generating Firebase token: {str(e)}")

    else:
        logger.error('Failed to get Spotify user info')
        raise HTTPException(
            status_code=400, detail="Failed to get Spotify user info")


def save_user_to_firestore(firebase_uid: str, user_data: dict):
    """
    Save new user data to Firestore.
    """
    try:
        db.collection('users').document(firebase_uid).set(user_data)
        logger.info(f"User {firebase_uid} saved to Firestore successfully.")
    except Exception as e:
        logger.error(f"Error saving user to Firestore: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error saving user to Firestore: {str(e)}")
