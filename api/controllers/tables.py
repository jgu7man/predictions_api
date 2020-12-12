from api.firebase_app import FirebaseApp
from rest_framework.views import APIView
from rest_framework.response import Response

import pandas as pd
import json

db = FirebaseApp.fs
st = FirebaseApp.st

class LoadTable(APIView):
    def get(self, request, format=None):
    
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.query_params['table']
            doc_ref = db.collection(u'tables').document(table_id)
        except:
            return Response({
                'message': 'La petición debe incluir el parámetro "table"',
                'status':500
            })
            

        # VALIDATE IS DOCUMENT IN FIRESTORE
        try:
            doc = doc_ref.get()
            result_data = doc.to_dict()
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
            'data': result_data,
            'product_list': json.loads(loadfile_result)
        }

        return Response({
            'message': 'ok', 
            'result': result,
            'status': 200
            })