import firebase_admin
from firebase_admin import credentials, firestore, storage
cred = credentials.Certificate({})


firebase_app = firebase_admin.initialize_app(cred, {'storageBucket': 'sales-predict.appspot.com'})

class FirebaseApp():
    fs = firestore.client()
    st = storage.bucket()
