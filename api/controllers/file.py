from api.firebase_app import FirebaseApp
from IPython.display import display
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
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
    
    # def get(self, request, format=None):
        
    #     return render(request, 'api/views/upload.html')
    #     doc_ref = db.collection(u'test').document(u'test_doc')
        
    #     try:
    #         doc = doc_ref.get()
    #         result = doc.to_dict()
    #         print(doc)
    #     except:
    #         print('no exite')
    #         result = 'no doc'
            
        
        
    
    # def post(self,request, format=None):
        
    #     return Response({
    #         'message':'ok'
    #     })