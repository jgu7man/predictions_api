from api.firebase_app import FirebaseApp
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from api.controllers.file import upload_file
from sklearn.svm import SVR
from pmdarima.arima import auto_arima

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  as mlp
import numpy as np
import locale
import datetime
import json

db = FirebaseApp.fs
st = FirebaseApp.st
mlp.style.use('seaborn')

class EstimatedPrediction(APIView):
    def post(self, request, format=None):
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.data['table']
        except: return Response({
                'message': 'La petición debe incluir el atributo "table" en el body',
                'status':500
            })
        
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.data['product']
        except: return Response({
                'message': 'La petición debe incluir el atributo "product" en el body',
                'status':500
            })
        
        # VALIDATE IS TEST SIZE    
        try: test_size = request.data['test_size']
        except: return Response({
                'message': 'La petición debe incluir el atributo "test_size" en el body',
                'status':500
            })
        
        # VALIDATE IS TEST SIZE    
        try: window_size = request.data['window_size']
        except: return Response({
                'message': 'La petición debe incluir el atributo "window_size" en el body',
                'status':500
            })
    
        # IMPORT FILE
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        doc = doc_ref.get()
        doc_URL = doc.to_dict()['timelineURL']
        product_name = doc.to_dict()['name']
        dataset = pd.read_json(doc_URL)
        dataset = dataset.fillna(method='ffill')
        
        predict_results = make_estimate_prediction(dataset, test_size, window_size, product_name)
        
        est_pred_imgURL = upload_file('api/uploads/',cloud_path, 'estimated_predict.jpg' )
        default_storage.delete('api/uploads/estimated_predict.jpg')
        est_pred_jsonURL = upload_file('api/uploads/',cloud_path, 'estimated_predict.json' )
        default_storage.delete('api/uploads/estimated_predict.json')
        
        result = {
            "predict_results": predict_results,
            "files": {
                "est_pred_imgURL": est_pred_imgURL,
                "est_pred_jsonURL": est_pred_jsonURL
            },
        }
        
        doc_ref.collection(u'predictions').document(u'estimated').set(result)
        
        return Response({
            "status":200,
            "message":"ok",
            "result":result
        })
        
        
        
        
def make_estimate_prediction(dataset, test_size, window_size, product_name):
    df_shift = dataset['unidades'].shift(1)
    df_mean_roll = df_shift.rolling(window_size).mean()
    df_std_roll = df_shift.rolling(window_size).std()
    df_mean_roll.name = "mean_roll"
    df_std_roll.name = "std_roll"
    df_mean_roll.index = dataset.index
    df_std_roll.index = dataset.index
    
    df_w = pd.concat([dataset['unidades'],df_mean_roll,df_std_roll],axis=1)

    df_w = df_w[window_size:]
    
    test_cant = int((dataset['unidades'].describe()['count'])*(test_size*.01))

    test = df_w[-test_cant:]
    train = df_w[:-test_cant]
    X_test = test.drop("unidades",axis = 1)
    y_test = test["unidades"]
    X_train = train.drop("unidades",axis = 1)
    y_train = train["unidades"]



    clf = SVR(gamma="scale")
    clf.fit(X_train, y_train)
    y_train_hat = pd.Series(clf.predict(X_train),index=y_train.index)
    y_test_hat = pd.Series(clf.predict(X_test),index=y_test.index)
    hat_groups = y_test_hat.groupby(pd.Grouper(freq='M'))

    total_predicted = np.sum(y_test_hat)
    mean_predicted = y_test_hat.describe()['mean']
    chart_data = {
        "train":y_train,
        "prediction":y_test_hat,
        "test":y_test,
    }
    predict_df = pd.DataFrame(data=chart_data)
    predict_df.to_json('api/uploads/estimated_predict.json', orient="columns")

    

    plt.figure(figsize=(12,6));
    plt.plot(y_train ,label='Datos de entrenamiento');
    plt.plot(y_test_hat,label='Predicción');
    plt.plot(y_test , label='Datos reales');
    plt.legend(loc='best')
    plt.title('Predicción de ventas ' + product_name)
    plt.savefig('api/uploads/estimated_predict.jpg')

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_test_hat)
    print('Margen de error: {}'.format(mse))
    
    
    result = {
        "total_predicted": int(total_predicted),
        "months_predicted": len(hat_groups),
        "avg_for_sell": float(mean_predicted),
        "error_mean" : mse,
    }
    
    return result



class ARIMAprediction(APIView):
    def post(self, request, format=None):
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.data['table']
        except: return Response({
                'message': 'La petición debe incluir el atributo "table" en el body',
                'status':500
            })
        
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.data['product']
        except: return Response({
                'message': 'La petición debe incluir el atributo "product" en el body',
                'status':500
            })
        
        # VALIDATE IS TEST SIZE   
        try: test_size = request.data['test_size']
        except: return Response({
                'message': 'La petición debe incluir el atributo "test_size" en el body',
                'status':500
            })
        
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        doc = doc_ref.get()
        files = doc.to_dict()['files']
        
        
        # VALIDATE DATASET
        try: doc_URL = files['sales_dates']
        except: return Response({
            'message':'Falta dataset de datos normalizados',
            'status': 404
        })
            
        
        dataset = pd.read_csv(doc_URL, header=None, index_col=0, parse_dates=True, squeeze=True)

        # split_point = math.floor( len(month_sales) *.80)
        # dataset, validation = month_sales[0:split_point], month_sales[split_point:]
        # print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
        # dataset.to_csv('dataset.csv', header=False)
        # validation.to_csv('validation.csv', header=False)
        
        groups = dataset.groupby(pd.Grouper(freq='M'))
        sells_avg = groups.describe()['count'].sum()/len(groups)

        
        Arima_model=auto_arima(dataset, start_p=1, start_q=1, max_p=8, max_q=8, start_P=0, start_Q=0, max_P=8, max_Q=8, m=12, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=30)
        
        predict = Arima_model.predict(n_periods=test_size)
        total_value_predictions = np.sum(predict)
        months_predicted = total_value_predictions / sells_avg
        
        plt.figure(figsize=(12,6))
        plt.plot(predict);
        plt.savefig('api/uploads/arima_prediction.jpg')
        
        predict_df  = pd.DataFrame(data=predict)
        predict_df.to_json('api/uploads/arima_prediction.json', orient="index")
        
        arima_pred_imgURL = upload_file('api/uploads/', cloud_path, 'arima_prediction.jpg')
        arima_pred_jsonURL = upload_file('api/uploads/', cloud_path, 'arima_prediction.json')
        
        
        predict_result = {
            "avg_for_sell": float(sells_avg),
            "total_predicted": int(total_value_predictions),
            "months_predicted":float(months_predicted),
            "files": {
                "arima_pred_img": arima_pred_imgURL,
                "arima_pred_json": arima_pred_jsonURL,
            }
        }
        
        doc_ref.collection(u'predictions').document(u'arima').set(predict_result)
        
        
        return Response({
            'status':200,
            'message': 'ok',
            'result': predict_result
        })
        
        