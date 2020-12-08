from api.firebase_app import FirebaseApp
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage


import pandas as pd
import matplotlib.pyplot as plt

import json


db = FirebaseApp.fs
st = FirebaseApp.st

class UploadFile(APIView):
    def get(self, request, format=None):
        return render(request, 'upload.html')
        
    
    def post(self, request, format=None):
        
        # LOAD FILE
        uploaded_file = request.FILES['dataset']
        filename = uploaded_file.name
        local_path = 'api/uploads/'+filename        
        # cloud_path = 'uploaded_files'
        df = pd.read_csv(uploaded_file,  decimal=".")
        df.to_csv(local_path)
        
        
        # FILTER THE LIST
        products_list = df[['codigo', 'descripcion']]
        products_list = products_list.drop_duplicates(subset=['codigo']).dropna()
        products_list['descripcion'] = products_list['descripcion'].str.strip()
        
        
        # GENERATE JSON RESULT
        loadfile_result = products_list.to_json(orient="table")
        count = products_list.describe()['codigo']['count']

        #STORAGE IN FIREBASE
        # bucket = st.bucket('tables')
        blob = st.blob('tables/'+filename)
        blob.upload_from_filename(local_path)
        blob.make_public()
        fileURL = blob.public_url
        
        result_data = {
            'total_count': int(count),
            'fileURL':fileURL,
            'file_name': filename,
        }
        
        doc_ref = db.collection(u'tables').document('test')
        doc_ref.set(result_data)
        
        result = {
            'data': result_data,
            'product_list': json.loads(loadfile_result)
        }
        
        
        
        
        default_storage.delete(local_path)
        return Response({'message': 'ok', 'result': result})



class LoadTable(APIView):
    def get(self, request, format=None):
    
        # VALIDATE THERE IS ID
        try:
            doc_id = request.query_params['id']
            doc_ref = db.collection(u'tables').document(doc_id)
        except:
            return Response({
                'message': 'La petición debe incluir un id en el body',
                'status':500
            })
            

        # VALIDATE IS DOCUMENT IN FIRESTORE
        try:
            doc = doc_ref.get()
            doc_URL = doc.to_dict()['fileURL']        
        except:
            return Response({
                'message': 'El archivo no exite o fue eliminado', 
                'status': 500
            })
        
        
        # FILTER THE LIST
        df = pd.read_csv(doc_URL,  decimal=".")
        products_list = df[['codigo', 'descripcion']]
        products_list = products_list.drop_duplicates(subset=['codigo']).dropna()
        products_list['descripcion'] = products_list['descripcion'].str.strip()
        
        
        # GENERATE JSON RESULT
        loadfile_result = products_list.to_json(orient="table")
        count = products_list.describe()['codigo']['count']
    
        result = {
            'total_count': int(count),
            'product_list': json.loads(loadfile_result)
        }

        return Response({
            'message': 'ok', 
            'result': result,
            'status': 200
            })