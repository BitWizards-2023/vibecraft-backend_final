import firebase_admin
from firebase_admin import credentials, firestore, auth

# Initialize Firebase app and Firestore client
cred = credentials.Certificate(
    'app/music-recommendation-38bc9-firebase-adminsdk-k5pj5-f80e3d943c.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
