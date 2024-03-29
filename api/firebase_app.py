import json
import firebase_admin
from firebase_admin import credentials, firestore, storage
# from firebase_admin.storage import storage

with open('api/sales-predict-firebase.json', 'r') as c:
    project_cred = json.load(c)

cred = credentials.Certificate(project_cred)
default_app = firebase_admin.initialize_app(cred, {'storageBucket':'sales-predict.appspot.com'})

class FirebaseApp:
    fs = firestore.client()
    st = storage