# from IPython.core.display import display
from api.firebase_app import FirebaseApp
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage

import pandas as pd
import json
import os
import io

db = FirebaseApp.fs
st = FirebaseApp.st

class UploadFile(APIView):
    def get(self, request, format=None):
        return render(request, 'upload.html')
    
    
    def post(self, request, format=None):
        
        # LOAD FILE
        try:uploaded_file = request.FILES['dataset']
        except: return Response({
            'message':'La petición no contiene archivo'
        }, status=400)
        
        global synonyms_cols
        try:request.data['synonyms']
        except:
            print('no sinónimos')
            synonyms_cols = {}
        else:
            synonyms_cols = request.data['synonyms']
        
        print(synonyms_cols)
        filename = uploaded_file.name
        filename = filename.replace(' ', '_')
        current_directory = os.path.abspath(os.path.dirname(__file__))+'/'
        # local_path = 'api/uploads/'+filename
        print('Archivo ok')
        
        # READ FILE
        df = pd.read_csv(uploaded_file,  decimal=".", header=0, thousands=r",")
        df = validate_file_struct(df, synonyms_cols)
        
        
        
        
        
        
        # FILTER THE LIST
        products_list = df[['Codigo', 'Descripcion']]
        products_list = products_list.drop_duplicates(subset=['Codigo']).dropna()
        products_list['Descripcion'] = products_list['Descripcion'].str.strip()

        # GENERATE JSON RESULT but don't upload in firebase storeage
        loadfile_result = products_list.to_json(orient="table")
        count = products_list.describe()['Codigo']['count']
        
        print('Lista creada')

        
        #STORAGE IN FIREBASE
        # Agregamos los datos válidos para obtener un id de firestore
        tables_ref = db.collection(u'tables')
        doc = tables_ref.add({
            'total_count': int(count),
            'file_name': filename,
        })
        doc_id = doc[1].id
        print('Id obtenido')
        
        
        # UPLOAD FILE CSV TO STORAGE
        cloud_path = 'tables/'+doc_id+'/'
        df.fillna(value=0, inplace=True)
        df_file = df.to_csv()
        fileURL = upload_file(cloud_path, filename, df_file)
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
        
        
        
        
        return Response({
            'status': 201,
            'message': 'ok', 
            'result': result
            }, status=201)



def validate_file_struct(df, synonyms_cols):
    try:df.drop(columns='Unnamed: 0')
    except:print("No hay columnas sin nombre")
    global code_des
    global cols_must_be 
    global needed_cols
    
    cols_must_be = ['Fecha', 'Unidades', 'Unitario Venta', 'Ventas', 'Costo Unitario', 'Total Costo', 'Margen', 'PorMargen', 'Codigo', 'Descripcion']
    needed_cols = ['Fecha', 'Unidades', 'Unitario Venta', 'Ventas', 'Costo Unitario', 'Total Costo', 'Margen', 'PorMargen']
    
    # revisar si las columnas vienen bien
    try: df[cols_must_be]
    except: 
        # Las columnas no vienen bien, se buscará si hay sinónimos
        if len(synonyms_cols) > 0:
            for key, value in synonyms_cols.items():
                # Se intenta cambiar las columnas, si no es posible, se rechaza la petición
                try:df = df.rename(columns={key:value})
                except: return Response({
                    'message': f'fallo al intentar cambiar columna {key} a {value}',
                    'status': 404
                }, status=404)
        
        # intenta revisar de nuevo las columnas
        try:df[cols_must_be]
        # si siguen sin estar bien, se revisa que la columna de codigo y descripción vengan fusionadas
        except:                         
            # primero revisamos que no haya error con las demás columnas
            try: df[needed_cols]
            except:
                # si hay error en las demás columnas revisamos en cuál y lo notificamos
                for col in needed_cols:
                    try: df[col]
                    except: return Response({
                            'message': f'no se encontró la columna {col}',
                            'status': 404
                        }, status=404)
            # si las columnas están bien, revisamos "CódigoInventario"
            else:
                try: df['CódigoInventario']
                except: return Response({
                    'message': 'archivo tampoco contiene columna "CódigoInventario"',
                    'status': 404
                }, status=404)
                # si la columna estaba fusionada, la separamos
                else:
                    code_des = df['CódigoInventario'].str.split(' ', n=1, expand=True )
                    df['Codigo'] = code_des[0]
                    df['Descripcion'] = code_des[1]
                    df.drop(columns=['CódigoInventario'], inplace=True)
    else: print('Archivo contiene columnas correctas')
    
    return df
        
        
def upload_file(cloud_path, filename, file):
    # print(cloud_path+filename)
    bucket = st.bucket()
    cloud_file = bucket.blob(cloud_path+filename)
    cloud_file.upload_from_string(file, content_type='application/octet-stream')
    cloud_file.make_public()
    # default_storage.delete(local_path+filename)
    return cloud_file.public_url



def upload_img(cloud_path, filename, file):
    img_data = io.BytesIO()
    file.savefig(img_data, format="jpg")
    img_data.seek(0)
    
    bucket = st.bucket()
    cloud_file = bucket.blob(cloud_path+filename)
    cloud_file.upload_from_file(img_data, content_type='image/jpg')
    cloud_file.make_public()
    return cloud_file.public_url
    