from api.firebase_app import FirebaseApp
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage

import pandas as pd
import json

db = FirebaseApp.fs
st = FirebaseApp.st
class UploadFile(APIView):
    def get(self, request, format=None):
        return render(request, 'upload.html')
        
    
    def post(self, request, format=None):
        
        # LOAD FILE
        try:
            uploaded_file = request.FILES['dataset']
        except: return Response({
            'status':400,
            'message':'La petición no contiene archivo'
        })
        
        print('Archivo ok')
        filename = uploaded_file.name
        local_path = 'api/uploads/'+filename        
        # cloud_path = 'uploaded_files'
        df = pd.read_csv(uploaded_file,  decimal=".")
        df.to_csv(local_path)
        
        
        print('Archivo leido')
        # # FILTER THE LIST
        try: products_list = df[['codigo', 'descripcion']]
        except: return Response({
            'status': 405,
            'message': 'La tabla no contiene columnas correctas'
        })
            
        products_list = products_list.drop_duplicates(subset=['codigo']).dropna()
        products_list['descripcion'] = products_list['descripcion'].str.strip()
        
        print('Lista creada')
        
        # # GENERATE JSON RESULT
        loadfile_result = products_list.to_json(orient="table")
        count = products_list.describe()['codigo']['count']

        #STORAGE IN FIREBASE
        
        tables_ref = db.collection(u'tables')
        doc = tables_ref.add({
            'total_count': int(count),
            'file_name': filename,
        })
        doc_id = doc[1].id
        
        print('Id obtenido')
        # bucket = st.bucket('tables')
        blob = st.blob('tables/'+doc_id+'/'+filename)
        blob.upload_from_filename(local_path)
        blob.make_public()
        fileURL = blob.public_url
        
        
        print('Archivo cargado a storage')
        result_data = {
            'total_count': int(count),
            'fileURL':fileURL,
            'file_name': filename,
            'doc_id':doc_id
        }
        
        print('Documento actualizado')
        tables_ref.document(doc_id).update(result_data)
        
        result = {
            'data': result_data,
            'product_list': json.loads(loadfile_result)
        }
        
        
        
        
        default_storage.delete(local_path)
        return Response({
            'status': 200,
            'message': 'ok', 
            'result': result
            })




        
        
def upload_file(local_path, cloud_path, filename):
        file = st.blob(cloud_path+filename)
        file.upload_from_filename(local_path+filename)
        file.make_public()
        return file.public_url